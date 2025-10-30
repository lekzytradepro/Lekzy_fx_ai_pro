FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "lekzy_fx_ai_pro.py"]
