# Source : https://openclassrooms.com/forum/sujet/cours-analysez-vos-donnees-textuelles
# Auteur : Christophe Naciri

import numpy as np
import os, sys, re, json, requests
import scrapy
 
from scrapy.crawler import CrawlerProcess, CrawlerRunner
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor


GENIUS_CLIENT_ACCESS_TOKEN = "cpPIVXItPPNHmuU7OdLsFHF2Z_yIqD0uS_QkOtRSnpnVOSiVAtcXBLbInIv4vABN"

 
class RapperSpider(scrapy.Spider):
    name = 'rappeurs français'

    start_urls = [
        'https://fr.wikipedia.org/wiki/Catégorie:Rappeur_français',
    ]
 
    def parse(self, response):
        for category_group in response.css('div.mw-category-group'):
            for letter_group in category_group.xpath('ul/li/a'):
                yield {
                    'rappeur': letter_group.css('::text').get()
                }
 
        next_page = response.xpath('//div[@id="mw-pages"]/a').attrib["href"]
        if next_page is not None:
            yield response.follow(next_page, self.parse)
 
    def stop(self):
        self.crawler.engine.close_spider(self, 'timeout')


def scrap_rappers_list():
    project_settings = get_project_settings()
    project_settings["FEEDS"] = {"rappeurs.json": {"format": "json"}}

    process_1 = CrawlerProcess(settings=project_settings)
    process_1.crawl(
        RapperSpider,
        start_urls=["https://fr.wikipedia.org/wiki/Catégorie:Rappeur_français"]
    )
    process_1.start(stop_after_crawl=True)


 
def scrap_rapper_songs(name):
    """ Fonction permettant d'aller lire sur genius.com la liste des chansons attribuées à ce rappeur """
    rx = requests.get(
        "https://api.genius.com/search?q=" + name.replace(' ','%20'),
        headers={"Authorization": "Bearer " + GENIUS_CLIENT_ACCESS_TOKEN}
    )

    # Ne garder que les morceaux dont le rappeur est l'unique artiste
    only_unique_artist_query_songs = [
        e for e in rx.json()["response"]["hits"]
        if e["result"]["artist_names"] == name
    ]

    if only_unique_artist_query_songs == []:
        return []
    
    artist_api_path = only_unique_artist_query_songs[0]["result"]["primary_artist"]["api_path"]
 
    rx = requests.get(
        "https://api.genius.com" + artist_api_path + "/songs?sort=popularity&per_page=50&page=1",
        headers={"Authorization":"Bearer " + GENIUS_CLIENT_ACCESS_TOKEN}
    )
 
    artist_songs = [
        e for e in rx.json()["response"]["songs"]
        if e["primary_artist"]["api_path"] == artist_api_path
    ]
 
    return list(map(lambda e: (e["title"], e["path"]), artist_songs))


# Définition d'un Spider Scrapy custom (pour extraire les paroles de N chansons d'un même rappeur)
class SongSpider(scrapy.Spider):
    name = 'paroles de chansons'
 
    def __init__(self, song_paths=[], *args, **kwargs):
        super(SongSpider, self).__init__(*args, **kwargs)
        self.start_urls = ["https://genius.com" + song_path for song_path in song_paths]
 
    def parse(self, response):
        """ Parser le fichier HTML de sorte à extraire les paroles """
        song_lyrics = []
        for lyrics_container in response.xpath("//div[@data-lyrics-container='true']"):
            for highlighted_verse in lyrics_container.xpath("a"):
                song_lyrics += highlighted_verse.xpath("span//text()").getall()
             
            song_lyrics += lyrics_container.xpath("text()").getall()
        
        yield {
            response.url.split("/")[-1] : song_lyrics
        }


def load_rappers_list():
    # Ici lire les rappeurs du fichier rappeurs.json
    rappers = []
    for dico in json.load(open('rappeurs.json')):
        rappers.append(dico["rappeur"])
    
    # Supprimer les parenthèses (exemple : "Abd al Malik (artiste)") + éventuels doublons + trier
    rappers = list(map(lambda name : re.sub(pattern="\((.*)\)", repl='', string=name).rstrip(), rappers))
    rappers = list(dict.fromkeys(rappers))
    rappers.sort()
    return rappers


def create_songs_dir():
    # Créer un répertoire "/paroles" pour contenir les fichiers des paroles
    if os.path.isdir("paroles") == False:
        path = os.path.join(os.curdir, "paroles")
        os.mkdir(path)



# pb : voir https://stackoverflow.com/questions/41495052/scrapy-reactor-not-restartable
# the wrapper to make it run more times
# def run_spider(spider): ...
def scrap_rapper_lyrics(rapper_name):
    create_songs_dir()

    rapper_songs = scrap_rapper_songs(rapper_name)
    if rapper_songs == []:
        print(f"No song found for rapper {rapper_name}")
        return
    
    project_settings = get_project_settings()
    project_settings["FEEDS"] = {"paroles/" + rapper_name + ".json": {"format": "json"}}
    
    runner = CrawlerRunner(settings=project_settings)
    
    rapper_song_paths = list(np.array(rapper_songs).T[1])
    runner.crawl(SongSpider, song_paths=rapper_song_paths)
    
    deferred = runner.join()
    
    deferred.addBoth(lambda _: reactor.stop())
    reactor.run()


def scrap_rappers_lyrics():
    for rapper_name in load_rappers_list():
        print(f"Extract {rapper_name} lyrics")
        scrap_rapper_lyrics(rapper_name)


"""
import os, json
 
os.system("python extract_rappers.py")
 
rappeurs = []
for dico in json.load(open('rappeurs.json')):
    rappeurs.append(dico["rappeur"])
 
for idx, rappeur_name in enumerate(rappeurs):
    os.system("python extract_single_rapper_songs.py "+str(idx))
"""