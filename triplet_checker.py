import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk
import pickle
import os


class TripletChecker():
    def __init__(self):
        # create window object
        self.window = tk.Tk()
        # image panel
        self.query_panel = None
        self.positive_panel = None
        self.negative_panel = None
        # triplet parameters
        self.triplet_pd = pd.read_csv('triplet.csv')
        if os.path.exists('triplet_check_point.pickle'):
            with open('triplet_check_point.pickle', 'rb') as f:
                self.triplet_index = pickle.load(f)
        else:
            self.triplet_index = 0

    def set_window_frame(self):
        """ window configuration - title & size """
        self.window.title('TRIPLET CHECKER')
        self.window.geometry('900x600+200+200')
        self.window.resizable(False, False)

    def set_image_frame(self):
        """ window configuration - image frames """
        # query
        query_frame = tk.LabelFrame(self.window, text='QUERY',
                                    width=230, height=230, font=('맑은고딕', 20, 'bold'))
        query_frame.pack(side='top', anchor='w', padx=170, pady=30)
        self.query_panel = tk.Label(query_frame, width=230, height=230)
        self.query_panel.pack()

        # positive
        positive_frame = tk.LabelFrame(self.window, text='POSITIVE',
                                       width=230, height=230, font=('맑은고딕', 20, 'bold'))
        positive_frame.place(x=170, y=330)
        self.positive_panel = tk.Label(positive_frame, width=230, height=230)
        self.positive_panel.pack()

        # negative
        negative_frame = tk.LabelFrame(self.window, text='NEGATIVE',
                                       width=230, height=230, font=('맑은고딕', 20, 'bold'))
        negative_frame.place(x=540, y=330)
        self.negative_panel = tk.Label(negative_frame, width=230, height=230)
        self.negative_panel.pack()

    def set_buttons(self):
        """ window configuration - buttons """
        # next 버튼 초록색 흰글씨
        next_button = tk.Button(self.window, text='next', command=self.next_query,
                                width=10, height=2, font=('맑은고딕', 20, 'bold'))
        next_button.place(x=500, y=80)

        # delete 버튼 빨간색 흰글씨
        delete_button = tk.Button(self.window, text='delete', command=self.delete_query,
                                  width=10, height=2, font=('맑은고딕', 20, 'bold'))
        delete_button.place(x=680, y=80)

        # save & exit 버튼
        save_button = tk.Button(self.window, text='save & exit', command=self.save_and_quit,
                                width=10, height=2, font=('맑은고딕', 20, 'bold'))
        save_button.place(x=580, y=220)

    def draw_image(self, path='sample.jpeg', type='query'):
        """ draw the image(query, pos, neg) """
        # select panel
        if type == 'query':
            target = self.query_panel
        elif type == 'positive':
            target = self.positive_panel
        else:
            target = self.negative_panel

        # load image and display in the panel
        img = ImageTk.PhotoImage(Image.open(path).resize((210, 210), Image.ANTIALIAS))
        target.configure(image=img)
        target.image = img

    def next_query(self, inc_idx=True):
        """ load next query image """
        # increase index
        if inc_idx:
            self.triplet_index += 1
        query_path = self.triplet_pd['query'][self.triplet_index]
        positive_path = self.triplet_pd['positive'][self.triplet_index]
        negative_path = self.triplet_pd['negative'][self.triplet_index]

        self.draw_image(path=query_path, type='query')
        self.draw_image(path=positive_path, type='positive')
        self.draw_image(path=negative_path, type='negative')

    def delete_query(self):
        """ delete current query image and load next image """
        # delete current dataframe info
        self.triplet_pd.drop(self.triplet_index, inplace=True)
        self.triplet_pd.reset_index(drop=True, inplace=True)
        print(self.triplet_pd)
        # next query
        self.next_query(inc_idx=False)

    def save_and_quit(self):
        """ save data and quit window """
        # save check-point
        with open('triplet_check_point.pickle', 'wb') as f:
            pickle.dump(self.triplet_index, f)

        # save to csv
        self.triplet_pd.to_csv('triplet.csv', index=False, mode='w')

        # destroy the window
        self.window.destroy()

    def run(self):
        """ main loop """
        # window setting
        self.set_window_frame()
        self.set_buttons()
        self.set_image_frame()

        # show image
        self.next_query(inc_idx=False)

        # execute program
        self.window.mainloop()


if __name__ == '__main__':
    program = TripletChecker()
    program.run()
