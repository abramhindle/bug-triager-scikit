FROM ubuntu:14.04

RUN apt-get update
RUN apt-get -y install curl

RUN curl -L https://get.rvm.io | bash -s stable;
ADD docker-env.sh ./
RUN ./docker-env.sh rvm install 1.9.2

RUN ./docker-env.sh gem install octokit

ADD . ./lda-chapter-tutorial
WORKDIR ./lda-chapter-tutorial

ENTRYPOINT ./docker-env.sh bash