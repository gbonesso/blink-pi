import kivy
kivy.require('2.1.0')  # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

class LoginScreen(GridLayout):

    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        self.cols = 2
        self.ear_left_label = Label(text='EAR L:')
        self.add_widget(self.ear_left_label)
        self.ear_right_label = Label(text='EAR R:')
        self.add_widget(self.ear_right_label)

class MyApp(App):

    def build(self):
        return LoginScreen()


if __name__ == '__main__':
    MyApp().run()
