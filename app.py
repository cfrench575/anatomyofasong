from flask import Flask, render_template, request
import markdown
import pandas as pd
import catboost as cb
from catboost import CatBoostClassifier, Pool
# https://www.etsy.com/listing/1596409299/musical-heartbeat-clipart-clipart-bundle?gpla=1&gao=1&&utm_source=google&utm_medium=cpc&utm_campaign=shopping_us_e-craft_supplies_and_tools-canvas_and_surfaces-stencils_templates_and_transfers-clip_art&utm_custom1=_k_CjwKCAjwoPOwBhAeEiwAJuXRh78nr1tJr9T8TNRJuqu7cTGH4Fa3TUAD_Xzf_LS9fDZzBD-4ESoL-BoC8LMQAvD_BwE_k_&utm_content=go_12573357968_118375525326_507851901085_aud-2079782229334:pla-295943621186_c__1596409299_5296766963&utm_custom2=12573357968&gad_source=1&gclid=CjwKCAjwoPOwBhAeEiwAJuXRh78nr1tJr9T8TNRJuqu7cTGH4Fa3TUAD_Xzf_LS9fDZzBD-4ESoL-BoC8LMQAvD_BwE
# https://www.etsy.com/listing/1692951475/game-headphone-clipart-dynamic-game?click_key=532385d3b2a6f6c7f19622f1d5f327a85b538885%3A1692951475&click_sum=feba0524&external=1&rec_type=ss&ref=pla_similar_listing_top-1&sts=1

app = Flask(__name__)

def get_ClassifierModel():
    model = cb.CatBoostClassifier(random_seed=42)
    model = model.load_model("music_model", format='cbm')
    return model

def get_prediction(model, values):
    # model.predict(values)
    preds = model.predict_proba(values)
    return preds[1]

@app.route('/')
def index():
    genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'brazil'
        , 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country'
        , 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep'
        , 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german'
        , 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle'
        , 'heavy-metal', 'hip-hop', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian'
        , 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay'
        , 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb', 'new-age', 'opera', 'pagode', 'party', 'piano'
        , 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'reggae', 'reggaeton'
        , 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter'
        , 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'study', 'swedish', 'synth-pop', 'tango', 'techno'
        , 'trance', 'trip-hop', 'turkish', 'world-music']
    keys =  ['None','C', 'C#/Bb', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    markdown_content_explainer = '''What makes a song popular? To find out, design your very own pop songs using custom machine learning model inputs to predict your song’s popularity.  For more information on the algorithm and training data (including a link to the python code) please visit this page [**here**](https://cfrench575.github.io/posts/spotify-random-forest/)'''
    html_content_explainer = markdown.markdown(markdown_content_explainer)
    return render_template('base.html', genres=genres, keys=keys, explainer_content=html_content_explainer)

@app.route('/get_predictions_from_inputs', methods=['POST'])
def get_predictions_from_inputs():
    keys_dict =  {'None':-1,'C':0, 'C#/Bb':1, 'D':2, 'D#/Eb':3, 'E':4, 'F':5, 'F#/Gb':6, 'G':7, 'G#/Ab':8, 'A':9, 'A#/Bb':10, 'B':11}
    dict_inputs = request.json
    ## code key numerically
    dict_inputs['key'] = keys_dict[dict_inputs['key']]
    ## convert to ms for model
    dict_inputs['duration_ms'] = dict_inputs['duration_ms']*60000
    print(dict_inputs)
    slider_values_dict = dict_inputs.pop('sliderValues')
    dict_inputs.update(slider_values_dict)

    model = get_ClassifierModel()
    values = [dict_inputs.get(x) for x in model.feature_names_]
    prediction = get_prediction(model, values)
    prediction_f = f"{round(prediction * 100, 1)}%"
    return {'prediction': prediction_f}, 200

if __name__ == '__main__':
    app.run(debug=True)