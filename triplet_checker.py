import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle
import pickle
import os


class TripletChecker():
    def __init__(self, path):
        # create window object
        self.window = tk.Tk()
        # image panel
        self.query_panel = None
        self.positive_panel = None
        self.negative_panel = None
        # triplet parameters
        self.path = path
        self.triplet_pd = pd.read_csv(self.path)
        if os.path.exists('triplet_check_point.pickle'):
            with open('triplet_check_point.pickle', 'rb') as f:
                self.triplet_index = pickle.load(f)
        else:
            self.triplet_index = 0
        # numbering label
        self.label = tk.Label(self.window, text='check ' + str(self.triplet_index) + 'th',
                              font=('맑은고딕', 17, 'bold'))
        self.label.place(x=40, y=50)
        # number entry
        self.entry_label = tk.Label(self.window, text='move to...', font=('맑은고딕', 13))
        self.entry_label.place(x=30, y=100)
        self.entry = tk.Entry(self.window, width=5)
        self.entry.place(x=30, y=120)

    def set_window_frame(self):
        """
        window configuration - title & size
        """
        self.window.title('TRIPLET CHECKER')
        self.window.geometry('900x600+200+200')
        self.window.resizable(False, False)
        # theme setting
        style = ThemedStyle(self.window)
        style.set_theme('breeze')

    def set_image_frame(self):
        """
        window configuration - image frames
        """
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
        """
        window configuration - buttons
        """
        # next 버튼
        next_button = tk.Button(self.window, text='-->', command=self.next_query,
                                width=10, height=2, font=('맑은고딕', 17, 'bold'))
        next_button.place(x=670, y=60)
        # prev 버튼
        prev_button = tk.Button(self.window, text='<--', command=self.prev_query,
                                width=10, height=2, font=('맑은고딕', 17, 'bold'))
        prev_button.place(x=500, y=60)
        # delete 버튼
        delete_button = tk.Button(self.window, text='Delete', command=self.delete_query,
                                  width=10, height=2, font=('맑은고딕', 17, 'bold'), fg='#fc0352')
        delete_button.place(x=590, y=140)
        # save & exit 버튼
        save_button = tk.Button(self.window, text='Save & Exit', command=self.save_and_quit,
                                width=10, height=2, font=('맑은고딕', 17, 'bold'))
        save_button.place(x=590, y=240)
        # index 수정 버튼
        edit_button = tk.Button(self.window, text='move', command=self.edit_index,
                                  width=5, height=1, font=('맑은고딕', 12))
        edit_button.place(x=100, y=125)

    def draw_image(self, path, type='query'):
        """
        draw the image(query, pos, neg)
        """
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
        """
        load next query image
        """
        # increase index
        if inc_idx:
            self.triplet_index += 1
        query_path = self.triplet_pd['query'][self.triplet_index]
        positive_path = self.triplet_pd['positive'][self.triplet_index]
        negative_path = self.triplet_pd['negative'][self.triplet_index]

        self.draw_image(path=query_path, type='query')
        self.draw_image(path=positive_path, type='positive')
        self.draw_image(path=negative_path, type='negative')

        # set label
        self.label.configure(text='check ' + str(self.triplet_index) + 'th')

    def prev_query(self):
        """
        load prev query image
        """
        if self.triplet_index is not 0:
            self.triplet_index -= 1
        self.next_query(inc_idx=False)

    def delete_query(self):
        """
        delete current query image and load next image
        """
        # delete current dataframe info
        self.triplet_pd.drop(self.triplet_index, inplace=True)
        self.triplet_pd.reset_index(drop=True, inplace=True)

        # next query
        self.next_query(inc_idx=False)

    def save_and_quit(self):
        """
        save data and quit window
        """
        # save check-point
        with open('triplet_check_point.pickle', 'wb') as f:
            pickle.dump(self.triplet_index, f)

        # save to csv
        self.triplet_pd.to_csv(self.path, index=False, mode='w')

        # destroy the window
        self.window.destroy()

    def edit_index(self):
        if self.entry.get() is '':
            return

        self.triplet_index = int(self.entry.get())
        self.next_query(inc_idx=False)
        self.clear_text()

    def clear_text(self):
        self.window.focus_set()
        self.entry.delete(0, 'end')

    def run(self):
        """
        main loop
        """
        # window setting
        self.set_window_frame()
        self.set_buttons()
        self.set_image_frame()

        # show image
        self.next_query(inc_idx=False)

        # execute program
        self.window.mainloop()


if __name__ == '__main__':
    program = TripletChecker('design_triplet.csv')
    program.run()
