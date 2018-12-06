# Run Docker locally:
* sudo docker build -t toxic_comment_attention . 
* sudo docker run -d -p 5001:5001 toxic_comment_attention
* curl -H "Content-Type: application/json" -d '{"data":"I hate you"}' http://localhost:5001

# Deploy on Kubernetes:
* * https://medium.com/analytics-vidhya/deploy-your-first-deep-learning-model-on-kubernetes-with-python-keras-flask-and-docker-575dc07d9e76
