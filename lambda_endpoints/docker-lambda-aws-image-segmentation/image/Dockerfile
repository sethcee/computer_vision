# get the base docker image
FROM public.ecr.aws/lambda/python:3.11
RUN yum install -y gcc gcc-c++
RUN pip install --upgrade pip

# copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# install specified requirements
RUN pip install -r requirements.txt

# copy all the files in ./src
COPY src/* ${LAMBDA_TASK_ROOT}

# this is going to set the default command to the handler function in main.py
# so the lambda function runs it when its called
CMD [ "main.handler"]