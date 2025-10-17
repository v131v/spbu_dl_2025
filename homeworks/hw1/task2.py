"""
Реализуйте базовые функции autograd. Можете вдохновиться видео от Andrej Karpathy.
Напишите класс, аналогичный предоставленному классу 'Element', который реализует основные операции autograd: сложение, умножение и активацию ReLU.
Класс должен обрабатывать скалярные объекты и правильно вычислять градиенты для этих операций посредством автоматического дифференцирования.
Плюсом будет набор предоставленных тестов, оценивающих правильность вычислений.
Большим плюсом будет, если тесты будут написаны с помощью unittest.
Можно использовать только чистый torch(без использования autograd и torch.nn).
Пример:
a = Node(2)
b = Node(-3)
c = Node(10)
d = a + b * c
e = d.relu()
e.backward()
print(a, b, c, d, e)
Output:  Node(data=2, grad=0)  Node(data=-3, grad=10)  Node(data=10, grad=-3)  Node(data=-28, grad=1)  Node(data=0, grad=1)
"""

import unittest


class Node:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Node(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def relu(self):
        val = self.data if self.data > 0 else 0
        out = Node(val, (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


class TestAutograd(unittest.TestCase):

    def test_addition(self):
        """Тест сложения"""
        a = Node(2)
        b = Node(3)
        c = a + b
        c.backward()

        self.assertEqual(c.data, 5)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_multiplication(self):
        """Тест умножения"""
        a = Node(2)
        b = Node(3)
        c = a * b
        c.backward()

        self.assertEqual(c.data, 6)
        self.assertEqual(a.grad, 3)
        self.assertEqual(b.grad, 2)

    def test_relu_positive(self):
        """Тест ReLU с положительным значением"""
        a = Node(5)
        b = a.relu()
        b.backward()

        self.assertEqual(b.data, 5)
        self.assertEqual(a.grad, 1)

    def test_relu_negative(self):
        """Тест ReLU с отрицательным значением"""
        a = Node(-5)
        b = a.relu()
        b.backward()

        self.assertEqual(b.data, 0)
        self.assertEqual(a.grad, 0)

    def test_combined_operations(self):
        """Тест из примера задания"""
        a = Node(2)
        b = Node(-3)
        c = Node(10)
        d = a + b * c
        e = d.relu()
        e.backward()

        self.assertEqual(a.data, 2)
        self.assertEqual(b.data, -3)
        self.assertEqual(c.data, 10)
        self.assertEqual(d.data, -28)
        self.assertEqual(e.data, 0)

        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        self.assertEqual(e.grad, 1)

    def test_complex_graph(self):
        """Тест сложного вычислительного графа"""
        a = Node(2)
        b = Node(3)
        c = Node(4)

        d = a * b  # 6
        e = d + c  # 10
        f = e.relu()  # 10
        g = f * Node(2)  # 20
        g.backward()

        self.assertEqual(g.data, 20)
        self.assertEqual(a.grad, 6)  # 2 * 3
        self.assertEqual(b.grad, 4)  # 2 * 2
        self.assertEqual(c.grad, 2)  # 2 * 1

    def test_multiple_uses(self):
        """Тест переменной, используемой несколько раз"""
        a = Node(2)
        b = a + a  # b = 2a = 4
        c = b * a  # c = b*a = 2a*a = 2a^2= 8
        c.backward()

        self.assertEqual(c.data, 8)
        # dc/da = d(2a^2)/da = 4a = 4*2 = 8
        self.assertEqual(a.grad, 8)

    def test_relu_zero(self):
        """Тест ReLU с нулевым значением"""
        a = Node(0)
        b = a.relu()
        b.backward()

        self.assertEqual(b.data, 0)
        self.assertEqual(a.grad, 0)

    def test_scalar_operations(self):
        """Тест операций со скалярами"""
        a = Node(5)
        b = a + 3  # 8
        c = b * 2  # 16
        c.backward()

        self.assertEqual(c.data, 16)
        self.assertEqual(a.grad, 2)


if __name__ == '__main__':
    unittest.main()
