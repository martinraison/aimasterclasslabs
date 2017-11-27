mkdir preprocessed
mogrify -path preprocessed -trim -alpha extract -thumbnail 28x28 -background black -gravity center -extent 28x28 -alpha off letters/*.png
