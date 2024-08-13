from collections import defaultdict
import os
import sys
import logging
import re
import traceback
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.data_manager import DataManager
from src.anthropic_text_generator import TextGenerator
from src.text_utils import  tokenize_text


def process_query(logger, query, retriever, generator, thread_history=None):
    """Process the query by retrieving documents and generating text, optionally using thread history."""
    logger.info('Processing query: %s', query)

    # Retrieve document IDs based on the query
    document_ids = retriever.retrieve_documents(query, os.getenv('EMBED_TEMPERATURE'))
    logger.info('Retrieved document IDs: %s', document_ids)

    # Get the content of the retrieved documents
    documents = []
    if len(document_ids) != 0:
        documents = [retriever.get_document_content(document_ids)]

    # Combine document content to create the context
    context = " ".join(documents)
    
    # Add thread history to context if available
    if thread_history:
        context = f"{thread_history} {context}"
    
    logger.info('Combined document and thread context: %s', context)

    # Tokenize the context
    token_chunks = tokenize_text(context)
    logger.info('Tokenized chunks: %s', token_chunks)

    # Generate text for each tokenized chunk
    generated_texts = generator.generate_text(query, token_chunks)
    logger.info('Generated texts: %s', generated_texts)

    return generated_texts

def remove_bot_mention(logger, text):
    """Remove bot mentions from the text."""
    cleaned_text = re.sub(r'<@U[A-Z0-9]+>', '', text).strip()
    logger.debug('Removed bot mentions: %s', cleaned_text)
    return cleaned_text

def fetch_thread_history(logger, client, channel, thread_ts):
    """Fetch all messages from a thread efficiently."""
    logger.info('Fetching thread history for channel %s and thread %s', channel, thread_ts)
    messages = []
    cursor = None

    while True:
        response = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=100,
            cursor=cursor
        )
        messages.extend(response['messages'])
        logger.debug('Fetched %d messages', len(response['messages']))

        cursor = response.get('response_metadata', {}).get('next_cursor')
        if not cursor:
            break

    logger.info('Total messages fetched: %d', len(messages))
    return messages

def handle_all_messages(event, say, logger, client, retriever, generator, thread_mentions):
    """Handle all messages where the app is mentioned or is part of the conversation."""
    logger.info('Handling message event: %s', event)
    
    text = event.get('text', '')
    channel = event['channel']
    thread_ts = event.get("thread_ts", None) or event["ts"]
    cleaned_text = remove_bot_mention(logger, text)

    if not text:
        logger.warning("No text found in the message.")
        return

    try:
        if "gracias" in cleaned_text.lower():
            logger.info('Received "gracias", stopping further responses in this thread.')
            thread_mentions[thread_ts] = False
            if "gracias" in text.lower():
                say(
                    text="¿Te sirvió mi respuesta?",
                    channel=channel,
                    thread_ts=thread_ts,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "¿Te sirvió mi respuesta?"
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Sí"
                                    },
                                    "value": "yes",
                                    "action_id": "response_yes"
                                },
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No"
                                    },
                                    "value": "no",
                                    "action_id": "response_no"
                                }
                            ]
                        }
                    ]
                )
                return
        # Recuperar el historial del hilo si ya ha sido mencionado
        thread_history = None
        if thread_mentions[thread_ts]:
            messages = fetch_thread_history(logger, client, channel, thread_ts)
            thread_history = ' '.join(message["text"] for message in messages)
        
        if not thread_mentions[thread_ts]:
            if "memorizar" in text.lower():
                if not thread_history:
                    messages = fetch_thread_history(logger, client, channel, thread_ts)
                    thread_history = ' '.join(message["text"] for message in messages)
                summary = generator.generate_summary(thread_history)
                retriever.store_documents([summary])
                result = "¡He memorizado el resumen del hilo!"
                logger.info('Stored summary of thread')
            else:
                result = process_query(logger, cleaned_text, retriever, generator, thread_history)
            
            # Marcar el hilo como uno donde el bot ha sido mencionado
            thread_mentions[thread_ts] = True

            logger.info('Sending result: %s', result)
            say(text=result, channel=channel, thread_ts=thread_ts)
        
        else:
            # Continuar respondiendo en el hilo si ya ha sido mencionado antes
            result = process_query(logger, cleaned_text, retriever, generator, thread_history)
            logger.info('Sending result: %s', result)
            say(text=result, channel=channel, thread_ts=thread_ts)

    except Exception as e:
        logger.error('Error while processing message: %s', e)
        logger.error(traceback.format_exc())
        say(text=str(e), channel=channel, thread_ts=thread_ts)

def handle_interactive_message(ack, event, say, logger):
    """Handle interactive messages from buttons."""
    ack()
    print(event)
    action_id = event['actions'][0]['action_id']
    user_id = event['user']['id']

    if action_id == "response_yes":
        # Enviar métrica a New Relic para respuesta positiva
        logger.info(f'User {user_id} found the response helpful.')
        #send_metric_to_new_relic("response_helpful", 1)
        say(text="¡Me alegra que te haya servido!", channel=event['channel']['id'], thread_ts=event['message']['ts'])

    elif action_id == "response_no":
        # Enviar métrica a New Relic para respuesta negativa
        logger.info(f'User {user_id} did not find the response helpful.')
        #send_metric_to_new_relic("response_not_helpful", 1)
        say(text="Lamento que no haya sido útil. ¡Estoy aquí para ayudar!", channel=event['channel']['id'], thread_ts=event['message']['ts'])

def send_metric_to_new_relic(metric_name, value):
    """Send a custom metric to New Relic."""
    # Aquí se debe implementar la lógica para enviar la métrica a New Relic.
    pass

def main():
    load_dotenv()

    logging.basicConfig(
        stream=sys.stdout,
        level=os.environ.get('LOG_LEVEL', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

    retriever = DataManager(logger, os.getenv('BEDROCK_EMBED_MODEL_ID'), os.getenv('CHROMA_DB_PATH'))
    generator = TextGenerator(logger, os.getenv('BEDROCK_MODEL_ID'))

    thread_mentions = defaultdict(bool)

    app.event("app_mention")(lambda event, say, logger: handle_all_messages(
        event, say, logger, app.client, retriever, generator, thread_mentions
    ))

    app.event("message")(
        lambda event, say, logger=logger: handle_all_messages(
            event, say, logger, app.client, retriever, generator, thread_mentions
        ) if thread_mentions[event.get("thread_ts", None) or event["ts"]] else None
    )

    app.action("response_yes")(
        lambda ack,say, body, logger: handle_interactive_message(ack, body, say, logger)
    )

    app.action("response_no")(
        lambda ack,say, body, logger: handle_interactive_message(ack, body, say, logger)
    )

    logger.info('Starting Slack app...')
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()

if __name__ == "__main__":
    main()