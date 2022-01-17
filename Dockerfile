FROM jhonatans01/python-dlib-opencv

WORKDIR /Face-Studio

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
