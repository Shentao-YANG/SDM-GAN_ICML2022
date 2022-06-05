for ((i=0;i<10;i+=1))
do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python toy_main.py --device "cuda:0" --seed $i --model 'cgan' &
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python toy_main.py --device "cuda:0" --seed $i --model 'dgan' &
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python toy_main.py --device "cuda:0" --seed $i --model 'ggan'
done
wait
printf '\n End of running... \n'
