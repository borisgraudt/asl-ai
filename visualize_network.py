import sys
import socket
import threading
import json
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

LAYER_NAMES = ['Input', 'Dense_1', 'Dense_2', 'Dense_3', 'Output']
LAYER_COLORS = [QtGui.QColor('#bfc7d5'), QtGui.QColor('#4a90e2'), QtGui.QColor('#4a90e2'), QtGui.QColor('#4a90e2'), QtGui.QColor('#ff9500')]
BG_COLOR = QtGui.QColor('#f8f8fa')

class LayerRect(QtWidgets.QGraphicsRectItem):
    def __init__(self, x, y, w, h, name, color):
        super().__init__(x, y, w, h)
        self.setBrush(QtGui.QBrush(color))
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.name = name
        self.activation = 0.0
        self.default_color = color
        self.setAcceptHoverEvents(True)
        self.anim = QtCore.QVariantAnimation()
        self.anim.valueChanged.connect(self.set_activation_color)

    def set_activation(self, value):
        value = float(np.clip(value, 0, 1))
        self.anim.stop()
        self.anim.setStartValue(self.activation)
        self.anim.setEndValue(value)
        self.anim.setDuration(300)
        self.anim.start()
        self.activation = value

    def set_activation_color(self, value):
        # Цвет от светло-серого к синему/оранжевому
        if self.name == 'Output':
            base = QtGui.QColor('#ffe5c2')
            active = QtGui.QColor('#ff9500')
        else:
            base = QtGui.QColor('#e6eaf2')
            active = QtGui.QColor('#4a90e2')
        color = QtGui.QColor(
            int(base.red() + (active.red()-base.red())*value),
            int(base.green() + (active.green()-base.green())*value),
            int(base.blue() + (active.blue()-base.blue())*value)
        )
        self.setBrush(QtGui.QBrush(color))

    def hoverEnterEvent(self, event):
        QtWidgets.QToolTip.showText(event.screenPos().toPoint(), f"{self.name}\nActivation: {self.activation:.3f}")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        QtWidgets.QToolTip.hideText()
        super().hoverLeaveEvent(event)

class NetworkScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QtGui.QBrush(BG_COLOR))
        self.layers = []
        self.arrows = []
        self.init_layers()
        self.init_arrows()
        self.init_legend()

    def init_layers(self):
        W, H = 120, 320
        spacing = 180
        y = 60
        for i, name in enumerate(LAYER_NAMES):
            x = 80 + i*spacing
            rect = LayerRect(x, y, W, H, name, LAYER_COLORS[i])
            self.addItem(rect)
            self.layers.append(rect)
            # Подпись
            label = QtWidgets.QGraphicsTextItem(name)
            label.setDefaultTextColor(QtGui.QColor('#222'))
            font = QtGui.QFont('San Francisco', 18, QtGui.QFont.Bold)
            label.setFont(font)
            label.setPos(x + W/2 - 40, y + H + 10)
            self.addItem(label)

    def init_arrows(self):
        for i in range(len(self.layers)-1):
            l1 = self.layers[i]
            l2 = self.layers[i+1]
            x1 = l1.rect().x() + l1.rect().width()
            y1 = l1.rect().y() + l1.rect().height()/2
            x2 = l2.rect().x()
            y2 = l2.rect().y() + l2.rect().height()/2
            path = QtGui.QPainterPath()
            path.moveTo(x1, y1)
            path.cubicTo(x1+40, y1, x2-40, y2, x2, y2)
            arrow = QtWidgets.QGraphicsPathItem(path)
            pen = QtGui.QPen(QtGui.QColor('#b0b8c9'), 8, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            arrow.setPen(pen)
            self.addItem(arrow)
            self.arrows.append(arrow)

    def init_legend(self):
        # Цветовая шкала
        grad = QtGui.QLinearGradient(0, 0, 120, 0)
        grad.setColorAt(0, QtGui.QColor('#e6eaf2'))
        grad.setColorAt(1, QtGui.QColor('#4a90e2'))
        rect = QtWidgets.QGraphicsRectItem(700, 420, 120, 24)
        rect.setBrush(QtGui.QBrush(grad))
        rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.addItem(rect)
        txt = QtWidgets.QGraphicsTextItem('Activation')
        txt.setDefaultTextColor(QtGui.QColor('#222'))
        font = QtGui.QFont('San Francisco', 14)
        txt.setFont(font)
        txt.setPos(700, 450)
        self.addItem(txt)

    def set_activations(self, acts):
        # acts: список np.array для каждого слоя (Dense_1, Dense_2, Dense_3, Output)
        # acts[0] - Dense_1, acts[1] - Dense_2, ...
        # Для Input слоя берём 0.0 (нет активации)
        acts_vis = [0.0]
        for act in acts:
            if isinstance(act, (list, np.ndarray)) and len(act) > 0:
                mean = float(np.mean(act))
            else:
                mean = 0.0
            acts_vis.append(mean)
        for i, rect in enumerate(self.layers):
            rect.set_activation(acts_vis[i])

class NetworkWindow(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Neural Network Visualization (Apple Style)')
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setScene(NetworkScene())
        self.setFixedSize(1100, 600)
        self.setSceneRect(0, 0, 1050, 550)

    def set_activations(self, acts):
        self.scene().set_activations(acts)

# --- Сервер для приёма активаций от main.py ---
def activation_server(window):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 50007))
    while True:
        data, _ = sock.recvfrom(65536)
        try:
            acts = json.loads(data.decode())
            acts = [np.array(a) for a in acts]
            QtCore.QMetaObject.invokeMethod(window, lambda acts=acts: window.set_activations(acts), QtCore.Qt.QueuedConnection)
        except Exception as e:
            print('Ошибка при приёме активаций:', e)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = NetworkWindow()
    window.show()
    threading.Thread(target=activation_server, args=(window,), daemon=True).start()
    sys.exit(app.exec_()) 