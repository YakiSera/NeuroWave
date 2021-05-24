import numpy

def save_all(error, edu_time, calc_time):
    error_txt = open("error.txt", "w")
    edu_txt = open("education_time.txt", "w")
    cacl_txt = open("calculation_time.txt", "w")

    for line in error:
        error_txt.write('{ln}\n'.format(ln=line))
    for line in edu_time:
        edu_txt.write('{ln}\n'.format(ln=line))
    for line in calc_time:
        cacl_txt.write('{ln}\n'.format(ln=line))
    error_txt.close()
    edu_txt.close()
    cacl_txt.close()