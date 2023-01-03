<h1 align='center'>JobPrepr - Model</h1>

**NOTE** - This is the backend model logic for the JobPrepr project, for the frontend user interface please visit https://github.com/HuwBarber/job-prepr-ui



<h2 align='center'> About the Project</h2>

<h3 align='center'>Background and Motivation</h3>

For many, practicing for interviews can be tough without somebody to give feedback. JobPrepr is an online tool which empowers users to practice for (non-technical) interviews by allowing them to record responses to questions randomly selected from a pool, and receive feedback on their facial expressions and sentiment of their speech.

Built in a 2 week sprint as part of Le Wagon's Data Science Bootcamp, in collaboration with:

- Andrei Danila - https://github.com/AndreiCDanila
- Martin Skalicky - https://github.com/skoooty
- Milica Šćepanović - https://github.com/scepanovicka

This repository contains a cut down, cleaned version of the codebase containing only what is necessary to run the application. If you'd like to view the original source code, please visit https://github.com/skoooty/job-prepr-model.

<h3 align='center'>Technologies and Frameworks Used - Models & Deployment</h3>

The model building portion of the project used a number of technologies and frameworks, including:

- Tensorflow Keras - CNN facial emotion recognition
- FastAPI
- Docker
- HuggingFace - Siebert for sentiment analysis
- GCP - Deployment & speech-to-text API

In addition to the technologies used by the frontend (linked at the top of this readme)

In addition, this project relies heavily on Google Cloud Services to run. The models were deployed as 2 Docker containers running on Cloud Run, one to handle the facial expressions and another to handle NLP. This allowed for concurrent processing to reduce a long wait time for users to get results.

At this time, these services are disabled due to run costs. If you would like a demonstration, feel free to contact me, or feel free to clone this repository and deploy to your own GCP project - a sample .env is provided.

Otherwise, take a look at our live demonstration of the our app using the link below (timestamp: 1:08:05)!

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/slVi4kWBiH4/0.jpg)](https://youtu.be/slVi4kWBiH4?t=4083)



