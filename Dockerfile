FROM jhonatans01/python-dlib-opencv

WORKDIR /Face-Studio

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
# ENTRYPOINT ["streamlit", "run"]
CMD ["gunicorn --bind 0.0.0.0:$PORT wsgi; streamlit run app.py"]
