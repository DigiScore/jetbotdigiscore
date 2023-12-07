# Collision Avoidance

There are 2 scripts required here to work with the DigiScore Jetbot ensemble:

    | audio.py
    | CA_Digiscore.ipynb

Place these together into the 'notebooks' folder of the 'jetbot'. They have been designed to run through JupyterLab interface
so the proprietary Jetbot disk image can be used. Follow instructions here https://jetbot.org/master/software_setup/sd_card.html

The following libraries need to be added at root level after the SD Card image has been installed AND the setup instructions have
 been completed.

1. Plug a screen into the Nano. 
Notice the Ubuntu UI has been disabled, so you should get a terminal window.

2. login id: jetbot
3. password: jetbot

4. check video camera is connected
~~~
ls -l /dev/video0
~~~

5. sudo apt-get update
password~: jetbot

6. get portaudio libraries
~~~
sudo apt-get install portaudio19-dev python-all-dev python3-all-dev
~~~

7. install pyaudio
~~~
python3 pip install pyaudio
~~~

8. run CA_Digiscore.py from webpage
Follow instructions here https://jetbot.org/master/examples/basic_motion.html
