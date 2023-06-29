# sdpointilism
Stablediffusion pointilism

Redering video can be achieved with the command

ffmpeg -framerate 30 -i '%d.png' -c:v libx264 -pix_fmt yuv420p out.mp4
