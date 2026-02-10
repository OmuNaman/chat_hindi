FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

COPY nano_hindi/ nano_hindi/
COPY chat.py .
COPY templates/ templates/

EXPOSE 7860

CMD ["python", "chat.py", "--host", "0.0.0.0", "--port", "7860", "--compile"]
