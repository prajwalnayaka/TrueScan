from functools import wraps
from authlib.integrations.flask_client import OAuth
from flask import session, redirect

def create_auth0_oauth(app):
    oauth = OAuth(app)
    
    auth0 = oauth.register(
        'auth0',
        client_id=app.config['AUTH0_CLIENT_ID'],
        client_secret=app.config['AUTH0_CLIENT_SECRET'],
        api_base_url=f'https://{app.config["AUTH0_DOMAIN"]}',
        access_token_url=f'https://{app.config["AUTH0_DOMAIN"]}/oauth/token',
        authorize_url=f'https://{app.config["AUTH0_DOMAIN"]}/authorize',
        jwks_uri=f'https://{app.config["AUTH0_DOMAIN"]}/.well-known/jwks.json',
        client_kwargs={'scope': 'openid profile email'}
    )
    
    return auth0

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'profile' not in session:
            return redirect('/')
        return f(*args, **kwargs)
    return decorated

def get_user_info():
    return session.get('profile')