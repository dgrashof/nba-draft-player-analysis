import requests
import pandas as pd
from bs4 import BeautifulSoup

ranking = []
name = []

for i in range(2000,2020):
    URL = 'https://www.basketball-reference.com/awards/recruit_rankings_'+str(i)+'.html'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    ranking.extend([i for i in range(1,101)])
    results = soup.find('tbody').find_all('tr')
    for i in range(110):
        for j,l in enumerate(results[i]):
            if j == 1:
                for m,n in enumerate(l):
                    if m ==0:
                        if 'href' in str(n):
                            if str(n).split('<')[1].split('>')[1] != 'RSCI':
                                name.append(str(n).split('<')[1].split('>')[1])
                        else:
                            if str(n) != 'RSCI':
                                name.append(str(n))




                                
pd.DataFrame({'hs_ranking':ranking,'name':name}).to_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/hs_rankings_df.csv',index = False)