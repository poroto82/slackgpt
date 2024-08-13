FROM python:3.12-slim

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

COPY . .

RUN pip install  --default-timeout=100 -r requirements.txt

# CMD predeterminado para ejecutar el Bot
CMD ["python", "bot.py"]
