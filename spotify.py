import os
from dataclasses import dataclass
from typing import List, Optional
import yaml
import requests
import pandas as pd

def load_creds():
    with open('creds.yml', 'r') as f:
        creds = yaml.load(f, Loader=yaml.FullLoader)
    return creds

CREDS = load_creds()
CLIENT_ID = CREDS['CLIENT_ID']
CLIENT_SECRET = CREDS['CLIENT_SECRET']

@dataclass
class Song:
    name: str
    artist: str
    album: str
    duration_ms: int
    popularity: int
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

    # add string representation
    def __str__(self):
        return f'{self.name} by {self.artist}'



def get_auth_token(client_id: str, client_secret: str) -> str:
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, data={
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })

    auth_response.raise_for_status()
    return auth_response.json()['access_token']


def get_playlist_tracks(playlist_id: str, auth_token: str) -> List[Song]:
    base_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {'Authorization': f'Bearer {auth_token}'}
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    tracks_data = response.json()['items']

    songs = []
    for track_data in tracks_data:
        track = track_data['track']
        audio_features = get_track_audio_features(track['id'], auth_token)
        song = Song(
            name=track['name'],
            artist=', '.join(artist['name'] for artist in track['artists']),
            album=track['album']['name'],
            duration_ms=track['duration_ms'],
            popularity=track['popularity'],
            danceability=audio_features['danceability'],
            energy=audio_features['energy'],
            key=audio_features['key'],
            loudness=audio_features['loudness'],
            mode=audio_features['mode'],
            speechiness=audio_features['speechiness'],
            acousticness=audio_features['acousticness'],
            instrumentalness=audio_features['instrumentalness'],
            liveness=audio_features['liveness'],
            valence=audio_features['valence'],
            tempo=audio_features['tempo'],
        )
        songs.append(song)

    return songs


def get_track_audio_features(track_id: str, auth_token: str) -> dict:
    base_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    headers = {'Authorization': f'Bearer {auth_token}'}
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    return response.json()


def write_songs_to_csv(songs: List[Song], output_file: str):
    df = pd.DataFrame([song.__dict__ for song in songs])
    df.to_csv(output_file, index=False)

def search_playlist(query: str, auth_token: str) -> str:
    search_url = f'https://api.spotify.com/v1/search?q={query}&type=playlist&limit=50'
    headers = {'Authorization': f'Bearer {auth_token}'}
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    playlists_data = response.json()['playlists']['items']

    print("Search results:")
    selected_index = None
    while selected_index is None:
        for j in range(1, len(playlists_data), 10):
            for i, playlist in enumerate(playlists_data[j:j+10]):
                print(f"{j+i + 1}. {playlist['name']} by {playlist['owner']['display_name']}")

            selected_index = input("Enter the number of the playlist you want to select: ")
            if selected_index == '':
                selected_index = None
            else:
                selected_index = int(selected_index) - 1
                # clear output
                print("\033c", end="")
                print(f'You selected {playlists_data[selected_index]["name"]} by {playlists_data[selected_index]["owner"]["display_name"]}')
    return playlists_data[selected_index]['id']

def search_playlist_interactive() -> str:
    OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query = input("Enter a keyword to search for a playlist: ")
    auth_token = get_auth_token(CLIENT_ID, CLIENT_SECRET)
    playlist_id = search_playlist(query, auth_token)
    songs = get_playlist_tracks(playlist_id, auth_token)
    output_file = os.path.join(OUTPUT_DIR, f'{playlist_id}.csv')
    write_songs_to_csv(songs, output_file)
    print(f'Playlist {playlist_id}')
    return playlist_id

def main():
    search_playlist_interactive()



if __name__ == '__main__':
    main()