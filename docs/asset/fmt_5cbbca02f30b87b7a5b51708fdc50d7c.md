
在运行器定义中，`HorovodRunner(np=2)`，将进程数指定为每个设置的两个（请参阅“设置分布式训练”），这将设置两个 `Standard_NC12` 工作人员 GPU。

训练例程完成后，请使用以下命令查看检查点目录：

```
ls -lrt  /dbfs/ml/CIFAR10DistributedDemo/train/1615074200.2146788/
```

您应该看到类似于这样的内容：

```
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-9.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-8.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-7.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-6.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-5.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-4.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-3.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-2.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-20.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-1.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-19.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-17.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-16.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-15.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-14.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-13.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-11.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-10.ckpt/
```

如果没有模型在先前的时代中有所改进，则会跳过一些检查点。最新的检查点代表具有最佳验证指标的模型。

# 总结

在本章中，您了解了在具有多个工作人员的环境中使分布式模型训练正常工作所需的内容。在数据并行框架中，有两种主要的分布式训练模式：异步参数服务器和同步 `allreduce`。如今，由于高性能加速器的普遍可用性，同步 `allreduce` 更受欢迎。

通过学习如何使用 `Databricks` GPU 集群执行两种类型的同步 `allreduce` API：`TensorFlow` 自己的`tf.distribute` API 和 `Uber` 的 `Horovod` API。`TensorFlow` 选项提供了最优雅和方便的使用方式，并且需要最少的代码重构，而 `Horovod` API 需要用户手动处理数据分片、分发管道、梯度聚合和平均以及模型检查点。这两种选项通过确保每个工作节点执行自己的训练，然后在每个训练步骤结束时，在所有工作节点之间同步和一致地更新梯度来执行分布式训练。这是分布式训练的标志。

恭喜！通过学习本章内容，您学会了如何使用云中的一组 GPU 训练具有分布式数据管道和分布式训练例程的深度学习模型。在下一章中，您将学习如何为推理服务一个 `TensorFlow` 模型。
