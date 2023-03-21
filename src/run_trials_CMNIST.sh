# python main.py -e append_test --color_var 0.02 
# python main.py -e append_test --color_var 0.02 
# python main.py -e append_test --color_var 0.04
# python main.py -e append_test --color_var 0.05




# for i in {1..2}
for i in {1..10}
do
    # start=`expr 100 \* $i - 100`
    # end=`expr 100 \* $i`
    python run_CMNIST.py
done


