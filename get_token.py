#!/usr/bin/python
import requests, json, getpass
from communication.communication import Communication
from communication.domain.messageValueObject import MessageVO
import ConfigParser
import io
import sys

CONFIG_FILE = "config.ini"
# CONFIG_FILE = "/home/pi/my_IDAS/config.ini"
TOKENS_URL = "https://ict4life-services.articatelemedicina.com/artica-uno-login/loginOauth/login/"
USERS_URL = "https://ict4life-services.articatelemedicina.com/artica-uno-login/admin/users/?role=USER"
PROFESSIONALS_USER_URL = "https://ict4life-services.articatelemedicina.com/artica-uno-login/admin/users/?professionals"
GET_PERSONS_FROM_USER = "https://ict4life-services.articatelemedicina.com/artica-uno-personas/admin/person/"
GET_PROFESSIONALS_FROM_USER = "https://ict4life-services.articatelemedicina.com/artica-uno-personas/admin/person/master/"
GET_PROFESSIONALS_FROM_USER2 = "https://ict4life-services.articatelemedicina.com/artica-uno-login/admin/profile/"

# GET_PROFESSIONALS_FROM_USER = "https://ict4life-services.articatelemedicina.com/artica-uno-personas/profile/professional/"
event_to_send = "fall down"
uuidUser = 'd20d7fc0-c0eb-4d49-8551-745bc149594e' #"c2c3ed00-c5fd-433f-9db7-4bf6dce11488"
band_name = "Alejandr's Band 98:71" #"MSFT Band UPM f6:65"
band_name_comp = "Alejandr's Band 98:71"#"MSFT Band UPM f6:65"


def real_report(band_name, event_to_send):
    try:
        with open(CONFIG_FILE, 'r+') as f:
            sample_config = f.read()
            config = ConfigParser.RawConfigParser(allow_no_value=True)
            config.readfp(io.BytesIO(sample_config))
            f.close()
    except (OSError, IOError) as e:
        print "file not found, please check config.ini exists and locate in myIDAS folder"
        print "archivo no encontrado, verifique que existe y dejelo en el directorio raiz myIDAS "
        sys.exit([arg])
    try:
        band_name_comp = config.get('user', 'band_name')
        # print "band  %s" % band_name_comp
        uuidUser = config.get('user', 'uuidUser')
    # print "user UuiD  %s" % uuidUser
    except ConfigParser.NoOptionError as e:
        print "error en configuracion, wrong configuration in : %s"  % e.option
        sys.exit([arg])
    # print "usuario"+USER

    if 1:#band_name_comp == band_name:
        try:
            USER2 = "hetra_es@ict4life.eu"
            PASSWORD2 = "123456789"
            PAYLOAD = "{\"userName\":\"" + USER2 + "\", \"password\":\"" + PASSWORD2 + "\"}"
            HEADERS = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            URL = TOKENS_URL
            # RESP = requests.post(URL, data=PAYLOAD, headers=HEADERS)
            RESP = requests.post(URL, data=PAYLOAD, headers=HEADERS)
            # print  RESP.json()
            new_token = RESP.json()["token"]
        # print new_token
        # new_uuidUser=RESP.json()["uuidUser"]
        except Exception as e:
            print "ERROR " + e.message
            return "connection with the server was not stablished, notifications will not be sent"
        try:
            HEADERS = {'Accept': 'application/json', 'Content-Type': 'application/json', 'access_token': new_token,
                       'uuidUser': uuidUser}
            RESP = requests.get(GET_PROFESSIONALS_FROM_USER2 + uuidUser + '/', headers=HEADERS)
            # print RESP.json()
            # uuid_To_send =  str(RESP.json()['persons'][0]['uuidPerson'])
            uuids_prof = []
            for i in RESP.json()['impersonationInfo']:
                uuid_To_send = str(i['uuidUser'])
                uuids_prof.append(uuid_To_send)
            # print uuid_To_send
        except Exception as e:
            print "ERROR " + e.message
            return "authentication failed, notifications will not be sent"
        try:
            # enviar datos professionals
            # for i in uuids_prof:
            #     message_content = MessageVO(title='Warning', text=str(event_to_send), uuid_to=i, priority="HIGH")
            #     com = Communication(message_content)
            #     print 'sending ....'
            #     com.send
            message_content = MessageVO(title='Warning', text=str(event_to_send), uuid_to=uuidUser, priority="HIGH")
            com = Communication(message_content)
            print 'sending  warning to caregivers and professionals! ....'
            com.send

            print '------ notification sent successfully!! -------'
        except Exception as e:
            print "ERROR " + e.message
    else:
        return "the band does not match any user in the platform"


############Get token of user


if __name__ == '__main__':
    valores = real_report(band_name, event_to_send)

#######################
# print RESP.text
# print "Token: "+RESP.json()["access"]["token"]["id"]
