# --*-- coding:utf-8 --*--

import matplotlib.pyplot as plt
import re

patt = re.compile(r'Loss: ([0-9\.]+)')

loss = []
with open('log-resnetsal', 'r') as f:
	for l in f:
		if 'Loss' in l:
			try:
				loss.append(float(patt.findall(l)[0]))
			except Exception as e:
				pass
plt.plot(loss)
plt.show()