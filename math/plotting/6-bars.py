#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# data
people = ['Farrah', 'Fred', 'Felicia']
x = np.arange(len(people))

apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

# stacked bars
plt.bar(x, apples, width=0.5, color='red', label='apples')
plt.bar(x, bananas, width=0.5, bottom=apples,
        color='yellow', label='bananas')
plt.bar(x, oranges, width=0.5, bottom=apples + bananas,
        color='#ff8000', label='oranges')
plt.bar(x, peaches, width=0.5, bottom=apples + bananas + oranges,
        color='#ffe5b4', label='peaches')

# labels and title
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')

# x-axis ticks
plt.xticks(x, people)

# y-axis range and ticks
plt.yticks(np.arange(0, 81, 10))
plt.ylim(0, 80)

# legend
plt.legend()

# display plot
# plt.show()
plt.savefig("plot.png")
