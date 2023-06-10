FROM apache/airflow:slim-2.6.1-python3.9

RUN pip install psycopg2-binary SQLAlchemy

COPY ./dist/* /tmp/

RUN pip install /tmp/*.whl



COPY ./scripts/* /
WORKDIR /app

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint.sh"]
