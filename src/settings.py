import os

redshift_username = os.environ['REDSHIFT_USERNAME']
redshift_password = os.environ['REDSHIFT_PASSWORD']

conn_args = {
    "drivername": "postgres",
    "username": redshift_username,
    "password": redshift_password,
    "host": "redshift.db.newsela.com",
    "port": "5439",
    "database": "analytics",
}