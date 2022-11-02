import os;

ar1=2.0;
ar2=4.0;

while ar2-ar1>0.01:
	ar = (ar1+ar2)/2;
	t = ar*500000;

	os.system("./tnnp -s left -r "+"{:0.3f}".format(ar)+" -q 1.0 -t "+"{:0.0f}".format(t)+" test");

	f = open('./test/log_'+"{:0.3f}".format(ar)+'_1.000.txt', 'r');
	str = f.read();
	data = str.split("\n");

	data.remove('');
	data1 = map(int, data);

	passes = map(lambda x: 1 if x>64 else 0, data1)
	going = map(lambda x: 1 if x>40 else 0, data1)

	gone = sum(map(lambda a,b: 1 if a-b==1 else 0,going[1:],going[:-1]))
	passed = sum(map(lambda a,b: 1 if a-b==1 else 0,passes[1:],passes[:-1]))

	if passed<gone-2: 
		ar2=ar;
	else:
		ar1=ar;

	print "gone/passed ",gone,passed
