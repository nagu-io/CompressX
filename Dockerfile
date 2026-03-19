FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt setup.sh ./

RUN chmod +x setup.sh && ./setup.sh

COPY . .

EXPOSE 8000

CMD ["uvicorn", "compressx.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
