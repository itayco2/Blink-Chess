"""The learning film (plan P11): one never-seen puzzle position across a run's 21 frames.

pick.py scores the 200 film candidates drawn from the Lichess band puzzles and short-lists 5 for
Itay (G9); extract.py runs every frame of the run once on the chosen position and writes film.json;
render.py turns film.json into one 4:5 1080x1350 master per language (en for the README, he for
LinkedIn) with a page driven frame by frame in Edge and piped to ffmpeg.
"""
