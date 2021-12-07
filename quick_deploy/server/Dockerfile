From ubuntu:latest

WORKDIR /app/src

RUN apt-get update && apt-get install python3-pip python3 -y && apt-get install wget -y && apt-get install zip -y

COPY ./req.txt /app

RUN pip3 --default-timeout=1000 install -r /app/req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./data /app/data

COPY ./src /app/src

RUN mkdir -p /app/src/models

RUN unzip /app/src/paraphrase-multilingual-mpnet-base-v2.zip -d /app/src/models/paraphrase-multilingual-mpnet-base-v2/


#RUN pip3 --default-timeout=1000 install Keras==2.3.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install futures==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install cryptography -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install aiofiles -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install gdown -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install pymilvus==2.0.0rc7 -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install PyMySQL -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install sentence_transformers==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip3 --default-timeout=1000 install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD python3 main.py         
