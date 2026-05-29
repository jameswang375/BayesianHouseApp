FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p sqlmodel

EXPOSE 7860

CMD ["marimo", "run", "BayesianWebApp.py", "--host", "0.0.0.0", "--port", "7860", "--no-token"]
