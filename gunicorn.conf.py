# Gunicorn configuration file
import multiprocessing

# Bind to the port defined by the environment variable or default to 10000
bind = "0.0.0.0:10000"

# Number of worker processes. For CPU-bound tasks (image processing), 
# usually (2 x num_cores) + 1 is recommended. 
# Since Render free tier is small, we'll stick to 2 workers to avoid OOM errors.
workers = 2

# Worker class. 'sync' is default and safer for CPU heavy tasks than 'gevent'.
worker_class = 'sync'

# Timeout. Defaults to 30s, which is too short for downloading the u2net model 
# or processing large images. Bump this up significantly.
timeout = 120 

# Log level
loglevel = 'info'

# Access log - records incoming HTTP requests
accesslog = '-'

# Error log - records Gunicorn errors
errorlog = '-'
