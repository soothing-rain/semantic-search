FROM python:3.9

WORKDIR /app/src

RUN apt-get update && apt-get install zip -y

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Alternatively: RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip3 install --upgrade pip

COPY ./requirements.txt /app

RUN pip3 --default-timeout=1000 install -r /app/requirements.txt

COPY ./data /app/data

COPY ./src /app/src

RUN mkdir -p /app/src/models

# For the first time: RUN wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/paraphrase-multilingual-mpnet-base-v2.zip
RUN unzip /app/src/paraphrase-multilingual-mpnet-base-v2.zip -d /app/src/models/paraphrase-multilingual-mpnet-base-v2/

CMD python3 main.py         
