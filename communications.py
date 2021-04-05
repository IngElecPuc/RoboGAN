import io
from PIL import Image
import copy

class TCPClient():
    def __init__(self, client, headersize, text_coding, bytesonbuffer):
        self.client = client
        self.headersize = headersize
        self.text_coding = text_coding
        self.bytesonbuffer = bytesonbuffer
        self.full_msg = b''
        self.connected = False
        self.last_msg = b''

    def Handshake(self):
        self.connected = False
        seeking = True
        reading_buffer = True
        while seeking:
            full_msg = b''
            new_msg = True
            while reading_buffer:
                msg = self.client.recv(self.bytesonbuffer) #buffer amount of bytes
                
                if new_msg:
                    msglen = int(msg[:self.headersize])
                    new_msg = False

                full_msg += msg

                if len(full_msg) - self.headersize == msglen:
                    if full_msg[self.headersize:].decode(self.text_coding) == "Handshake?":
                        seeking = False
                        reading_buffer = False
                        self.connected = True


        response = "Off course my friend!"
        msg = f'{len(response):>{self.headersize}}' + response
        self.client.send(bytes(msg, self.text_coding))
        print("Handshake successful")

        return self.connected

    def Receive(self):
        seeking = True
        reading_buffer = True
        while seeking:
            full_msg = b''
            new_msg = True
            reading_buffer = True
            while reading_buffer:
                msg = self.client.recv(self.bytesonbuffer) #buffer amount of bytes

                if new_msg:
                    msglen = int(msg[:self.headersize])
                    new_msg = False

                full_msg += msg

                if msg.find(b'<EOS>') > 0:
                    reading_buffer = False
                    seeking = False
                    self.last_msg = copy.copy(full_msg)

                if msg.find(b'<EOC>') > 0:
                    reading_buffer = False
                    seeking = False
                    self.connected = False
                    print("Connection shut down from the client")
                    
        imgs = []
        traj = []
        objs = []
        
        if self.connected:
            while True:
                byteimg = b''
                header = full_msg[:self.headersize]
                if header == b'':
                    break
                imglen = int(header.decode(self.text_coding))
                byteimg = full_msg[self.headersize:(self.headersize + imglen)]
                full_msg = full_msg[(self.headersize + imglen):]
                if byteimg.find(b'<SON>') == -1 and byteimg.find(b'<EOC>') == -1:
                    img = Image.open(io.BytesIO(byteimg))                
                    imgs.append(img)        
                else:
                    break

            header = full_msg[:self.headersize]
            trajlen = int(header.decode(self.text_coding))
            traj = full_msg[self.headersize:(self.headersize + trajlen)].decode(self.text_coding)
            full_msg = full_msg[(self.headersize + trajlen):]

            traj_ = copy.copy(traj)
            traj_ = traj_.split("|")
            traj = []
            for tri in traj_:
                tri = tri.split(";")
                tri = [float(tri[0]), float(tri[1]), float(tri[2])]
                traj.append(tri)
            
            header = full_msg[:self.headersize]
            objslen = int(header.decode(self.text_coding))
            objs = full_msg[self.headersize:(self.headersize + objslen)].decode(self.text_coding)
            full_msg = full_msg[(self.headersize + objslen):] 

            objs_ = copy.copy(objs)
            objs_ = objs_.split("|")
            objs = []
            for obj in objs_:
                obj = obj.split(";")
                obj = [float(obj[0]), float(obj[1])]
                objs.append(obj)
        
        return imgs, traj, objs

    def Send(self, prediction):
        prediction = prediction.cpu().detach().numpy()
        msg = ''
        for (i, line) in enumerate(prediction):
            for (j, num) in enumerate(line):
                msg += str(num)
                if j < 2:
                    msg += ';'
            if i < 11:
                msg += '|'
        msg = f'{len(msg):>{self.headersize}}' + msg
        self.client.send(bytes(msg, self.text_coding))