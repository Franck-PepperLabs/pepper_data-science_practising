"""Scraping"""

from bs4 import BeautifulSoup
import requests
import pandas as pd


def get_brazil_states():
    """Get the dataframe containing the list of states in Brazil
    from Wikipedia.
    """
    # get the HTML content
    url = 'https://en.wikipedia.org/wiki/Federative_units_of_Brazil#List'
    response = requests.get(url)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # turn it into dataframe
    return pd.read_html(str(table))[0]


def scrap_brazil_municipalities():
    """Get the dataframe containing the list of municipalities in Brazil
    from Wikipedia.
    """
    # get the HTML content
    url = 'https://en.wikipedia.org/wiki/List_of_municipalities_of_Brazil'
    response = requests.get(url)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    data_list = []
    for table in soup.find_all('table', {'class': 'wikitable sortable'}):
        # Find the h2 (Brazil's region)
        #      and h3 (Brazil's state) elements preceding the table
        h2 = table.find_previous('h2')
        h3 = table.find_previous('h3')
        # Extract the text of the h2 and h3 element
        region_name = h2.text[:-len('[edit]')]
        state_name_and_code = h3.text[:-len('[edit]')]
        state_code = state_name_and_code[-3:-1]
        state_name = state_name_and_code[:-5]
        # Read the table and turn it into dataframe
        data = pd.read_html(str(table))[0]
        # Insert region and state data to the dataframe
        data.insert(0, 'CO', state_code)
        data.insert(0, 'State', state_name)
        data.insert(0, 'Region', region_name)
        # Add the dataframe to the list
        data_list += [data]

    # merge into a unique dataframe
    all = pd.concat(data_list, axis=0, ignore_index=True)

    # Add a column 'is_state_capital' and remove (State Capital) from name
    is_state_capital = all.Municipality.str.endswith('(State Capital)')
    all['is_state_capital'] = is_state_capital
    fixed_names = all.Municipality.str[:-len(' (State Capital)')]
    all.loc[is_state_capital, 'Municipality'] = fixed_names
    return all