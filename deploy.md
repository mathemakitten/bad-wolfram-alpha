* Set up nginx 
* Change nginx conf

`sudo service nginx start`

`sudo service nginx configtest`

`sudo service nginx restart`

`gunicorn --workers 2 --worker-class gevent --bind 0.0.0.0:5000 app:app --timeout 30000`