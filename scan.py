import os;

D1=0.25;
D2=0.25;
#N = 7;
D = D1;
dD = 0.5;
#dD = (D2-D1)/N; 

while D<=D2:

	print D
	
	os.system("./tnnp -s none -D "+"{:0.3f}".format(D)+" -t 10000000 test");
	os.system("cp ./test/backup.bin ./test/init_"+"{:0.3f}".format(D)+".bin");
	
	os.system("./tnnp -s left -i ./test/init_"+"{:0.3f}".format(D)+".bin -t 1000000 -D "+"{:0.3f}".format(D)+" test");
	os.system("cp ./test/log_front.txt ./test/log_"+"{:0.3f}".format(D)+".txt");
	os.system("cp ./test/log_AP.txt ./test/logAP_"+"{:0.3f}".format(D)+".txt");
	D += dD;
