FROM python:3.6

USER root

WORKDIR /app

ADD . /app

ENV CFLAGS="$CFLAGS -g0 -Wl,--strip-all -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib"
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8050

ENV NAME World

CMD ["python", "index.py"]