import cv2
import numpy as np

AZUL = (255,0,0)
ROJO = (0,0,255)
VERDE = (0,255,0)
VIOLETA = (255,0,255)
AMARILLO = (0,255,255)
CYAN = (255,255,0)
BLANCO = (255,255,255)

COLORES = [AZUL,VERDE,ROJO,VIOLETA,AMARILLO,CYAN,BLANCO]


	

image_path = './data/imagen_prueba.png'
output_path = './data/imagen_prueba_out.png'
output_txt = './data/regiones.txt'


img = cv2.imread(image_path)
scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)

dim = (width, height)

  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
aux_img = img.copy()


ix = -1
iy = -1
drawing = False
cant_regiones = 0
coordenadas = [0,0,0,0]


def on_mouse(event, x, y, flags, params):
	global ix,iy,drawing,img,aux_img,coordenadas

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix = x
		iy = y

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing:
			img = aux_img.copy()
			cv2.rectangle(img, pt1=(ix,iy), pt2=(x, y),color=(0,0,0),thickness=2)


	elif event == cv2.EVENT_LBUTTONUP:
		if drawing:
			drawing = False
			img = aux_img.copy()
			cv2.rectangle(img, pt1=(ix,iy), pt2=(x, y),color=COLORES[cant_regiones],thickness=2)
			coordenadas = [max(min(ix,x)/width,0),
						   max(min(iy,y)/height,0),
						   min(max(ix,x)/width,1),
						   min(max(iy,y)/height,1)]

def fill_alpha(coordenadas):
	global img
	x = int(coordenadas[0]*width)
	y = int(coordenadas[1]*height)
	w = int(coordenadas[2]*width - x)
	h = int(coordenadas[3]*height - y)
	sub_img = img[y:y+h, x:x+w]
	white_rect = np.ones(sub_img.shape, dtype=np.uint8)
	for c in range(3):
		white_rect[:,:,c] = white_rect[:,:,c]*COLORES[cant_regiones][c]

	res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

	# Putting the image back to its position
	img[y:y+h, x:x+w] = res

cv2.namedWindow('real image')
cv2.setMouseCallback('real image', on_mouse, 0)

regiones = []
print("Seleccione la region de interes con el mouse, y presione la tecla Y para confirmar la seleccion")
while True:
	cv2.imshow('real image', img)
	tecla = cv2.waitKey(10)
	if tecla== 27:
		break
	elif tecla==ord('y'):
		print(f"Region {cant_regiones+1} agregada. Si desea finalizar, presione Enter")
		fill_alpha(coordenadas)
		cant_regiones+=1
		aux_img = img.copy()
		regiones.append(coordenadas)
	elif tecla==13:
		cv2.imwrite(output_path, aux_img)
		a_file = open(output_txt, "w")
		for reg in regiones:
			np.savetxt(a_file, reg, fmt='%.2f',newline=" ")
		a_file.close()

		print("Los cambios han sido guardados")
		break