#creando un robotsin SAM para chatear con el/ella
#Idioma en Español

from googletrans import Translator, constants
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import numpy as np

print('Descargando archivos de Inteligencia artificial')
mitoken = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
samybot = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")


ecom_translate = Translator()



def text_usuario_sam_español(oracion):
	usuario_palabra = str(oracion)
	translation = ecom_translate.translate(f"{usuario_palabra}", dest = 'en')
	#print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
	return str(translation.text)


def text_sam_usuario_español(oracion):
	usuario_palabra = str(oracion)
	translation = ecom_translate.translate(f"{usuario_palabra}", dest = 'es')
	#print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
	return str(translation.text)

def cargando ():

	time.sleep(.4)
	print('.')
	time.sleep(.4)
	print('..')
	time.sleep(.4)
	print('...')


def preguntar_sam(palabra):
	oracion = text_usuario_sam_español(palabra)
	entradaBlender = mitoken([oracion], return_tensors='pt')
	ids_respuesta = samybot.generate(**entradaBlender)
	respuesta = mitoken.batch_decode(ids_respuesta)
	respuesta = respuesta[0].replace('<s>','').replace('</s>','')
	respuesta = text_sam_usuario_español(respuesta)


	return respuesta



print("hola bienvenido")

print("Escribe algo para iniciar la conversación")


veces = 0
while veces != 7:
	mioracion = input ('Escribe aqui: ')
	respuets = preguntar_sam(mioracion)
	print('')
	print('')
	print (respuets)
	veces = veces + 1
