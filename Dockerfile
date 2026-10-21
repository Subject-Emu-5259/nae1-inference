FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
RUN pip install "tokenizers==0.21.0" "huggingface_hub==0.26.0" "jinja2==3.1.4" "safetensors"
COPY nae1_model.py nae1_config.py server.py ./
RUN mkdir -p /app/data
EXPOSE 10000
CMD ["python", "server.py"]
