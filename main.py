from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse
import pandas as pd
import re

df = pd.read_csv('malicious_phish.csv')

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # IPv6
    return 1 if match else 0

def url_length(url):
    return len(url)

def hostname_length(url):
    return len(urlparse(url).netloc)

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def count_dot(url):
    return url.count('.')

def count_www(url):
    return url.count('www')

def count_atrate(url):
    return url.count('@')

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    return 1 if match else 0

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def count_letters(url):
    return sum(c.isalpha() for c in url)

df['having_ip'] = df['url'].apply(lambda x: having_ip_address(x))
df['url_length'] = df['url'].apply(lambda x: url_length(x))
df['hostname_length'] = df['url'].apply(lambda x: hostname_length(x))
df['suspicious_words'] = df['url'].apply(lambda x: suspicious_words(x))
df['count_dot'] = df['url'].apply(lambda x: count_dot(x))
df['count_www'] = df['url'].apply(lambda x: count_www(x))
df['count_atrate'] = df['url'].apply(lambda x: count_atrate(x))
df['no_of_dir'] = df['url'].apply(lambda x: no_of_dir(x))
df['no_of_embed'] = df['url'].apply(lambda x: no_of_embed(x))
df['shortening_service'] = df['url'].apply(lambda x: shortening_service(x))
df['count_https'] = df['url'].apply(lambda x: count_https(x))
df['count_http'] = df['url'].apply(lambda x: count_http(x))
df['count_ques'] = df['url'].apply(lambda x: count_ques(x))
df['count_hyphen'] = df['url'].apply(lambda x: count_hyphen(x))
df['count_equal'] = df['url'].apply(lambda x: count_equal(x))
df['count_letters'] = df['url'].apply(lambda x: count_letters(x))

X = df[['having_ip', 'url_length', 'hostname_length', 'suspicious_words', 'count_dot', 'count_www', 'count_atrate',
        'no_of_dir', 'no_of_embed', 'shortening_service', 'count_https', 'count_http', 'count_ques', 'count_hyphen',
        'count_equal', 'count_letters']]
y = df['type'] 

lb_make = LabelEncoder()
y = lb_make.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=5)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Test Accuracy: {accuracy * 100:.2f}%')