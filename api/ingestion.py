import pandas as pd
import requests

def get_page(page, base_url = 'http://127.0.0.1:5000/api/v1/data?page='):
    """ Return a pandas dataframe. Page of the database: page"""
    api_request = base_url + str(page)
    return pd.DataFrame(requests.get(api_request).json())

def get_pages(pages, base_url = 'http://127.0.0.1:5000/api/v1/data?page='):
    """ Return a pandas dataframe. Page of the database: pages"""
    df = pd.DataFrame()

    for page in pages:
        df = df.append(get_page(page, base_url=base_url))

    return df

if __name__ == '__main__':
    # an example of using the api
    save_res = False
    pages = range(0, 5)
    # make sure the api is running: python api/app.py
    df = get_pages(pages)
    print(df)


