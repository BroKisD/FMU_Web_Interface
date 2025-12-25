FROM python:3.12

RUN mkdir /app
WORKDIR /app 

COPY requirements.txt .
 
# Install any necessary dependencies 
RUN pip install --upgrade pip \
&& apt-get clean \
&& apt-get update \
&& apt-get install libgl1 -y \
&& pip install --no-cache-dir -r requirements.txt  

# Copy the rest of the application code into the container 
COPY . . 
   
#Expose the port the app runs on 
EXPOSE 8080

ENV HOST=0.0.0.0
ENV PORT=8080

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"] 