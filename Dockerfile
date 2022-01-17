FROM jhonatans01/python-dlib-opencv

WORKDIR /Face-Studio

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
