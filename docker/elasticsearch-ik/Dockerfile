ARG ES_VERSION=8.19.10
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}

ARG ES_VERSION=8.19.10

RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch \
    https://get.infini.cloud/elasticsearch/analysis-ik/${ES_VERSION}
