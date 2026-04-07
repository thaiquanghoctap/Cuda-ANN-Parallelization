#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void read_data(const char *filename, float **X, int nrows, int ncols, int label){
    FILE *file = fopen(filename, "r");
    if (file == NULL){
        printf("Không thể mở tệp");
        exit(EXIT_FAILURE);
    }

    *X = (float *)malloc(nrows * ncols * sizeof(float));
    if (*X == NULL){
        printf("Không thể cấp phát bộ nhớ");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    int index = 0;
    for (int i = 0; i < nrows; i++){
        for (int j = 0; j < ncols; j++){
            if (fscanf(file, "%f", &(*X)[index]) != 1){
                printf("Lỗi đọc dữ liệu từ tệp tại vị trí (%d, %d)\n", i, j);
                fclose(file);
                free(*X);
                exit(EXIT_FAILURE);
            }
            if(label == 0)
                (*X)[index] = (*X)[index] / 255.0;
            index++;
        }
    }

    fclose(file);
}

void to_onehot(float *Y, float *Y_onehot, int n_data, int n_class)
{
    for (int i = 0; i < n_data; i++)
    {
        for (int j = 0; j < n_class; j++)
        {
            Y_onehot[i * n_class + j] = 0.0;
        }
        Y_onehot[i * n_class + (int)(Y[i])] = 1.0;
    }
}

void print_matrix(float* A, int nrows, int ncols)
{
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            printf("%.5f  ", A[i * ncols + j]);
        }
        printf("\n");
    }
}

void log_training_details(int epoch, int epochs, double epoch_time, int n_samples, double loss, double accuracy, const char* filename) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        printf("Không thể mở file %s\n", filename);
        return;
    }

    // Ghi tiêu đề nếu file trống
    if (ftell(file) == 0) {
        fprintf(file, "Epoch,Total Epochs,Epoch Time (s),Time per Step (ms),Loss,Accuracy\n");
    }

    // Tính thời gian mỗi bước (step) tính bằng mili giây
    double time_per_step = (epoch_time * 1000) / n_samples;

    // Ghi chi tiết quá trình huấn luyện
    fprintf(file, "%d,%d,%.6f,%.6f,%.6f,%.6f\n", epoch + 1, epochs, epoch_time, time_per_step, loss, accuracy);

    fclose(file);
}

void log_details(const char* id, int n_epochs
                    , double matmul_time, double add_bias_time
                    , double relu_time, double softmax_time
                    , double transpose_time, double matsub_time
                    , double sum_rows_time, double relu_derivative_time
                    , const char* filename) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        printf("Không thể mở file %s\n", filename);
        return;
    }

    // Ghi tiêu đề nếu file trống
    if (ftell(file) == 0) {
        fprintf(file, "ID,Total Epochs,Matmul Time (s),Add Bias Time (s),ReLU Time (s),Softmax Time (s),Transpose Time (s),Matsub Time (s),Sum Rows Time (s),ReLU Derivative Time (s)\n");
    }


    // Ghi chi tiết quá trình huấn luyện
    fprintf(file, "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
            id, n_epochs,
            matmul_time, add_bias_time, relu_time, softmax_time, 
            transpose_time, matsub_time, sum_rows_time, relu_derivative_time);

    fclose(file);
}

void log_accuracy(const char* id, int n_epochs, int correct_predictions, int n_samples, double f1_score) {
    const char* filename = "./log/accuracy-log.csv";
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        printf("Không thể mở file %s\n", filename);
        return;
    }

    // Ghi tiêu đề nếu file trống
    if (ftell(file) == 0) {
        fprintf(file, "ID,Total Epochs,Accuracy,F1-Score\n");
    }

    // Ghi chi tiết độ chính xác và F1-score
    fprintf(file, "%s,%d,%.6f,%.6f\n", id, n_epochs, (float) correct_predictions / n_samples, f1_score);

    fclose(file);
}

void matmul(float* A, float* B, float* C, int nrows_A, int ncols_A, int ncols_B)
{
    for (int i = 0; i < nrows_A; i++)
    {
        for (int j = 0; j < ncols_B; j++)
        {
            C[i * ncols_B + j] = 0;
            for (int k = 0; k < ncols_A; k++)
            {
                C[i * ncols_B + j] += A[i * ncols_A + k] * B[k * ncols_B + j];
            }
        }
    }
}

void add_bias(float* Z, float*b, int nrows, int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            Z[i * ncols + j] += b[j];
        }
    }
}

void relu(float* Z, float* A, int nrows, int ncols)
{
    for(int i = 0; i < nrows * ncols; i++)
    {
        if(Z[i] > 0)
            A[i] = Z[i];
        else
            A[i] = 0;
    }
}

void softmax(float* Z, float* A, int nrows, int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        float sum_exp = 0.0;
        for (int j = 0; j < ncols; j++)
        {
            A[i * ncols + j] = expf(Z[i * ncols + j]);
            sum_exp += A[i * ncols + j];
        }

        for (int j = 0; j < ncols; j++)
        {
            A[i * ncols + j] /= sum_exp;
        }
    }
}

void init_weight(float* W, int nrows, int ncols)
{
    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            W[i * ncols + j] = ((float)rand() / RAND_MAX) * 0.1 - 0.05;
        }
    }
}

float compute_loss(float* A3, float* Y, int nrows, int ncols)
{
    float loss = 0.0;

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            float y_ij = Y[i * ncols + j];
            float a3_ij = A3[i * ncols + j];

            if (y_ij > 0.0)
            {
                loss += y_ij * log(fmax(a3_ij, 1e-9));
            }
        }
    }

    return (-1) * (loss / nrows);
}

float compute_accuracy(float* A3, float* Y, int nrows, int ncols)
{
    int correct_predictions = 0;

    for (int i = 0; i < nrows; i++)
    {
        int predicted_class = 0;
        int true_class = 0;

        // Tìm lớp dự đoán (chỉ số có giá trị lớn nhất trong A3)
        float max_prob = A3[i * ncols];
        for (int j = 1; j < ncols; j++){
            if (A3[i * ncols + j] > max_prob)
            {
                max_prob = A3[i * ncols + j];
                predicted_class = j;
            }
        }

        // Tìm lớp đúng (chỉ số có giá trị 1 trong Y)
        for (int j = 0; j < ncols; j++)
        {
            if (Y[i * ncols + j] == 1.0)
            {
                true_class = j;
                break;
            }
        }

        // So sánh dự đoán với nhãn đúng
        if (predicted_class == true_class)
        {
            correct_predictions++;
        }
    }

    // Tính độ chính xác
    return ((float)correct_predictions / nrows);
}

void matsub(float* A, float* B, float* C, int nrows, int ncols)
{
    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            C[i * ncols + j] = A[i * ncols + j] - B[i * ncols + j];
        }
    }
}

void sum_rows(float* dZ, float* db, int nrows, int ncols)
{
    for (int j = 0; j < ncols; j++) // Duyệt qua từng cột
    {
        db[j] = 0.0;
        for (int i = 0; i < nrows; i++) // Duyệt qua từng hàng
        {
            db[j] += dZ[i * ncols + j]; // Cộng phần tử dZ[i][j] vào db[j]
        }
    }
}

void transpose(float* A, float* A_T, int nrows, int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            A_T[j * nrows + i] = A[i * ncols + j]; // A_T[j][i] = A[i][j]
        }
    }
}

void relu_derivative(float *Z, float *dA, float *dZ, int nrows, int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            dZ[i * ncols + j] = (Z[i * ncols + j] > 0) ? dA[i * ncols + j] : 0.0;
        }
    }
}

void update_weights(float *W, float *dW, int nrows, int ncols, float lr)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            W[i * ncols + j] -= lr * dW[i * ncols + j];
        }
    }
}

void clip_gradient(float *gradient, int nrows, int ncols)
{
    float threshold = 30.0;

    float norm = 0.0;

    for (int i = 0; i < nrows * ncols; i++)
    {
        norm += gradient[i] * gradient[i];
    }
    norm = sqrt(norm);

    if (norm > threshold)
    {
        float scale = threshold / norm;
        for (int i = 0; i < nrows * ncols; i++)
        {
            gradient[i] *= scale;
        }
    }
}

void train(float* A0, float* Y, float* W1, float* b1, float* W2, float* b2, float* W3, float* b3,
      int n_samples, int input_layer_nodes, int hidden_layer1_nodes, int hidden_layer2_nodes, int output_layer_nodes,
      float lr, int n_epochs){
    const char* filename = "./log/host-training-log.csv";
    printf("\nSTARTING TRAIN ...\n");
    // MALLOC FOR FORWARD
    float* Z1 = (float*)malloc(n_samples * hidden_layer1_nodes * sizeof(float)); // N x 256
    float* A1 = (float*)malloc(n_samples * hidden_layer1_nodes * sizeof(float)); // N x 256

    float* Z2 = (float*)malloc(n_samples * hidden_layer2_nodes * sizeof(float)); // N x 128
    float* A2 = (float*)malloc(n_samples * hidden_layer2_nodes * sizeof(float)); // N x 128

    float* Z3 = (float*)malloc(n_samples * output_layer_nodes * sizeof(float)); // N x 10
    float* A3 = (float*)malloc(n_samples * output_layer_nodes * sizeof(float)); // N x 10

    // MALLOC FOR BACKWARD
    // float* dA3
    float* dZ3 = (float*)malloc(n_samples * output_layer_nodes * sizeof(float)); // N x 10
    float* db3 = (float*)malloc(1 * output_layer_nodes * sizeof(float)); // 1 x 10
    float* dW3 = (float*)malloc(hidden_layer2_nodes * output_layer_nodes * sizeof(float)); // 128 x 10

    float* dA2 = (float*)malloc(n_samples * hidden_layer2_nodes * sizeof(float)); // N x 128
    float* dZ2 = (float*)malloc(n_samples * hidden_layer2_nodes * sizeof(float)); // N x 128
    float* db2 = (float*)malloc(1 * hidden_layer2_nodes * sizeof(float)); // 1 x 128
    float* dW2 = (float*)malloc(hidden_layer1_nodes * hidden_layer2_nodes * sizeof(float)); // 256 x 128

    float* dA1 = (float*)malloc(n_samples * hidden_layer1_nodes * sizeof(float)); // N x 256
    float* dZ1 = (float*)malloc(n_samples * hidden_layer1_nodes * sizeof(float)); // N x 256
    float* db1 = (float*)malloc(1 * hidden_layer1_nodes * sizeof(float)); // 1 x 256
    float* dW1 = (float*)malloc(input_layer_nodes * hidden_layer1_nodes * sizeof(float)); // 784 x 256

    float* A2_T = (float*)malloc(hidden_layer2_nodes * n_samples * sizeof(float));
    float* W3_T = (float*)malloc(output_layer_nodes * hidden_layer2_nodes * sizeof(float));
    float* A1_T = (float*)malloc(hidden_layer1_nodes * n_samples * sizeof(float)); // 256 x N
    float* W2_T = (float*)malloc(hidden_layer2_nodes * hidden_layer1_nodes * sizeof(float));
    float* A0_T = (float*)malloc(input_layer_nodes * n_samples * sizeof(float)); // 784 x N

    // TIME
    float matmul_time = 0.0;
    float add_bias_time = 0.0;
    float relu_time = 0.0;
    float softmax_time = 0.0;

    float transpose_time = 0.0;
    float matsub_time = 0.0;
    float sum_rows_time = 0.0;
    float relu_derivative_time = 0.0;

    clock_t start, end;
    clock_t forward_start, forward_end;
    clock_t backward_start, backward_end;
    clock_t update_start, update_end;

    for(int epoch = 0; epoch <= n_epochs; epoch++){
        // FORWARD
        forward_start = clock();

        start=clock(); matmul(A0, W1, Z1, n_samples, input_layer_nodes, hidden_layer1_nodes); // Z1 = A0 x W1
        end=clock(); 
        if (epoch < n_epochs)
            matmul_time += (float)(end - start) / CLOCKS_PER_SEC;

        start=clock(); add_bias(Z1, b1, n_samples, hidden_layer1_nodes); // Z1 = Z1 + b1
        end=clock(); 
        if (epoch < n_epochs)
            add_bias_time += (float)(end - start) / CLOCKS_PER_SEC;

        start=clock(); relu(Z1, A1, n_samples, hidden_layer1_nodes); // A1 = relu(Z1)
        end=clock(); 
        if (epoch < n_epochs)
            relu_time += (float)(end - start) / CLOCKS_PER_SEC;
        /*-------------------*/
        start=clock(); matmul(A1, W2, Z2, n_samples, hidden_layer1_nodes, hidden_layer2_nodes); // Z2 = A1 x W2
        end=clock(); 
        if (epoch < n_epochs)
            matmul_time += (float)(end - start) / CLOCKS_PER_SEC;

        start=clock(); add_bias(Z2, b2, n_samples, hidden_layer2_nodes); // Z2 = Z2 + b2
        end=clock(); 
        if (epoch < n_epochs)
            add_bias_time += (float)(end - start) / CLOCKS_PER_SEC;

        start=clock(); relu(Z2, A2, n_samples, hidden_layer2_nodes); // A2 = relu(Z2)
        end=clock(); 
        if (epoch < n_epochs)
            relu_time += (float)(end - start) / CLOCKS_PER_SEC;
        /*-------------------*/
        start=clock(); matmul(A2, W3, Z3, n_samples, hidden_layer2_nodes, output_layer_nodes); // Z3 = A2 x W3
        end=clock(); 
        if (epoch < n_epochs)
            matmul_time += (float)(end - start) / CLOCKS_PER_SEC;

        start=clock(); add_bias(Z3, b3, n_samples, output_layer_nodes); // Z3 = Z3 + b3
        end=clock(); 
        if (epoch < n_epochs)
            add_bias_time += (float)(end - start) / CLOCKS_PER_SEC;

        start=clock(); softmax(Z3, A3, n_samples, output_layer_nodes); // A3 = softmax(Z3)
        end=clock(); 
        if (epoch < n_epochs)
            softmax_time += (float)(end - start) / CLOCKS_PER_SEC;

        forward_end = clock();

        // Loss
        float loss = compute_loss(A3, Y, n_samples, output_layer_nodes);
        float accuracy = compute_accuracy(A3, Y, n_samples, output_layer_nodes);

        // Time
        float forward_time = (forward_end - forward_start) / CLOCKS_PER_SEC;
        float backward_time = 0.0;
        float update_time = 0.0;
        float epoch_time = 0.0;

        if (epoch < n_epochs){
            // BACK PROPAGATION
            backward_start = clock();

            start=clock(); matsub(A3, Y, dZ3,n_samples, output_layer_nodes); // dZ3 (N x 10) = A3 - Y
            end=clock(); matsub_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); sum_rows(dZ3, db3, n_samples, output_layer_nodes); // db3 (1 x 10) = sum(dZ3, axis=0)
            end=clock(); sum_rows_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); transpose(A2, A2_T, n_samples, hidden_layer2_nodes); // A2_T (128 x N)
            end=clock(); transpose_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); matmul(A2_T, dZ3, dW3, hidden_layer2_nodes, n_samples, output_layer_nodes); // dW3 (128 x 10) = A2_T (128 x N) x dZ3 (N x 10)
            end=clock(); matmul_time += (float)(end - start) / CLOCKS_PER_SEC;
            /*-------------------*/
            start=clock(); transpose(W3, W3_T, hidden_layer2_nodes, output_layer_nodes); // W3_T (10 x 128)
            end=clock(); transpose_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); matmul(dZ3, W3_T, dA2, n_samples, output_layer_nodes, hidden_layer2_nodes); // dA2 (N x 128) = dZ3 (N x 10) x W3_T (10 x 128)
            end=clock(); matmul_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); relu_derivative(Z2, dA2, dZ2, n_samples, hidden_layer2_nodes); // dZ2 = dA2 . relu'(Z2)
            end=clock(); relu_derivative_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); sum_rows(dZ2, db2, n_samples, hidden_layer2_nodes); // db2 (1 x 128) = sum(dZ2, axis=0)
            end=clock(); sum_rows_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); transpose(A1, A1_T, n_samples, hidden_layer1_nodes); // A1_T (256 x N)
            end=clock(); transpose_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); matmul(A1_T, dZ2, dW2, hidden_layer1_nodes, n_samples, hidden_layer2_nodes); // dW2 (256 x 128) = A1_T (256 x N) x dZ2 (N x 128)
            end=clock(); matmul_time += (float)(end - start) / CLOCKS_PER_SEC;
            /*-------------------*/
            start=clock(); transpose(W2, W2_T, hidden_layer1_nodes, hidden_layer2_nodes); // W2_T (128 x 256)
            end=clock(); transpose_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); matmul(dZ2, W2_T, dA1, n_samples, hidden_layer2_nodes, hidden_layer1_nodes); // dA1 (N x 256) = dZ2 (N x 128) x W2_T (128 x 256)
            end=clock(); matmul_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); relu_derivative(Z1, dA1, dZ1, n_samples, hidden_layer1_nodes); // dZ1 = dA1 . relu'(Z1)
            end=clock(); relu_derivative_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); sum_rows(dZ1, db1, n_samples, hidden_layer1_nodes); // db1 (1 x 256) = sum(dZ1, axis=0)
            end=clock(); sum_rows_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); transpose(A0, A0_T, n_samples, input_layer_nodes); // A0_T (784 x N)
            end=clock(); transpose_time += (float)(end - start) / CLOCKS_PER_SEC;

            start=clock(); matmul(A0_T, dZ1, dW1, input_layer_nodes, n_samples, hidden_layer1_nodes); // dW1 (784 x 256) = A0_T (784 x N) x dZ1 (N x 256)
            end=clock(); matmul_time += (float)(end - start) / CLOCKS_PER_SEC;

            backward_end = clock();

            // UPDATE
            update_start = clock();
            clip_gradient(db3, 1, output_layer_nodes);
            clip_gradient(dW3, hidden_layer2_nodes, output_layer_nodes);

            clip_gradient(db2, 1, hidden_layer2_nodes);
            clip_gradient(dW2, hidden_layer1_nodes, hidden_layer2_nodes);

            clip_gradient(db1, 1, hidden_layer1_nodes);
            clip_gradient(dW1, input_layer_nodes, hidden_layer1_nodes);


            update_weights(W3, dW3, hidden_layer2_nodes, output_layer_nodes, lr);
            update_weights(b3, db3, 1, output_layer_nodes, lr);

            update_weights(W2, dW2, hidden_layer1_nodes, hidden_layer2_nodes, lr);
            update_weights(b2, db2, 1, hidden_layer2_nodes, lr);

            update_weights(W1, dW1, input_layer_nodes, hidden_layer1_nodes, lr);
            update_weights(b1, db1, 1, hidden_layer1_nodes, lr);

            update_end = clock();

            // TIME
            backward_time = (backward_end - backward_start) / CLOCKS_PER_SEC;
            update_time = (update_end - update_start) / CLOCKS_PER_SEC;

        }
        epoch_time = (forward_time + backward_time + update_time);

        printf("\nEpoch %d/%d [====================] - %.2fms/step - loss: %.6f - accuracy: %.6f\n", epoch + 1, n_epochs,(epoch_time * 1000) / n_samples, loss, accuracy);
        printf("\t\tEpoch Time: %.5fs - Forward Time: %.5fs - Backward Time: %.5fs - Update Time: %.5fs\n", forward_time + backward_time + update_time, forward_time, backward_time, update_time);
        log_training_details(n_epochs, epoch, epoch_time, n_samples, loss, accuracy, filename);
}

    // FREE
    free(Z1); free(A1); free(Z2); free(A2); free(Z3); free(A3);

    free(dZ3); free(db3); free(dW3);
    free(dA2); free(dZ2); free(db2); free(dW2);
    free(dA1); free(dZ1); free(db1); free(dW1);

    free(A2_T);
    free(W3_T); free(A1_T);
    free(W2_T); free(A0_T);

    printf("\nTimes of each computation:\n");
    printf("\tMatmul Time: %.5f seconds\n", matmul_time);
    printf("\tAdd Bias Time: %.5f seconds\n", add_bias_time);
    printf("\tReLU Time: %.5f seconds\n", relu_time);
    printf("\tSoftmax Time: %.5f seconds\n", softmax_time);
    printf("\tTranspose Time: %.5f seconds\n", transpose_time);
    printf("\tMatsub Time: %.5f seconds\n", matsub_time);
    printf("\tSum Rows Time: %.5f seconds\n", sum_rows_time);
    printf("\tReLU Derivative Time: %.5f seconds\n", relu_derivative_time);

    log_details("Host training", n_epochs, matmul_time, add_bias_time, relu_time, softmax_time, transpose_time, matsub_time, sum_rows_time, relu_derivative_time, "./log/time-details.csv");
}

void test(float* A0, float* Y, float* W1, float* b1, float* W2, float* b2, float* W3, float* b3,
          int n_samples, int input_layer_nodes, int hidden_layer1_nodes, int hidden_layer2_nodes, int output_layer_nodes, int n_epochs)
{
    printf("\n\nSTARTING TEST ...\n");

    // Cấp phát bộ nhớ cho forward propagation
    float* Z1 = (float*)malloc(n_samples * hidden_layer1_nodes * sizeof(float));
    float* A1 = (float*)malloc(n_samples * hidden_layer1_nodes * sizeof(float));
    float* Z2 = (float*)malloc(n_samples * hidden_layer2_nodes * sizeof(float));
    float* A2 = (float*)malloc(n_samples * hidden_layer2_nodes * sizeof(float));
    float* Z3 = (float*)malloc(n_samples * output_layer_nodes * sizeof(float));
    float* A3 = (float*)malloc(n_samples * output_layer_nodes * sizeof(float));

    // Ma trận confusion
    int true_positives[output_layer_nodes];
    int false_positives[output_layer_nodes];
    int false_negatives[output_layer_nodes];
    memset(true_positives, 0, sizeof(true_positives));
    memset(false_positives, 0, sizeof(false_positives));
    memset(false_negatives, 0, sizeof(false_negatives));

    int correct_predictions = 0;

    // Forward propagation
    matmul(A0, W1, Z1, n_samples, input_layer_nodes, hidden_layer1_nodes);
    add_bias(Z1, b1, n_samples, hidden_layer1_nodes);
    relu(Z1, A1, n_samples, hidden_layer1_nodes);

    matmul(A1, W2, Z2, n_samples, hidden_layer1_nodes, hidden_layer2_nodes);
    add_bias(Z2, b2, n_samples, hidden_layer2_nodes);
    relu(Z2, A2, n_samples, hidden_layer2_nodes);

    matmul(A2, W3, Z3, n_samples, hidden_layer2_nodes, output_layer_nodes);
    add_bias(Z3, b3, n_samples, output_layer_nodes);
    softmax(Z3, A3, n_samples, output_layer_nodes);

    // Predict and update confusion matrix
    for (int i = 0; i < n_samples; i++) {
        int true_label = -1, predicted_label = -1;
        float max_prob = -1.0;

        for (int j = 0; j < output_layer_nodes; j++) {
            if (Y[i * output_layer_nodes + j] == 1.0) true_label = j;
            if (A3[i * output_layer_nodes + j] > max_prob) {
                max_prob = A3[i * output_layer_nodes + j];
                predicted_label = j;
            }
        }

        if (true_label == predicted_label) {
            correct_predictions++;
            true_positives[true_label]++;
        } else {
            false_positives[predicted_label]++;
            false_negatives[true_label]++;
        }
    }

    // Calculate F1-score
    float macro_f1 = 0.0;
    for (int i = 0; i < output_layer_nodes; i++) {
        float precision = 0.0;
        if ((true_positives[i] + false_positives[i]) > 0) {
            precision = (float)true_positives[i] / (true_positives[i] + false_positives[i]);
        }

        float recall = 0.0;
        if ((true_positives[i] + false_negatives[i]) > 0) {
            recall = (float)true_positives[i] / (true_positives[i] + false_negatives[i]);
        }

        float f1_score = 0.0;
        if ((precision + recall) > 0.0) {
            f1_score = 2.0 * (precision * recall) / (precision + recall);
        }
        macro_f1 += f1_score;
    }
    macro_f1 /= output_layer_nodes;

    // Kết quả
    printf("Test - Accuracy: %.6f%%\n", (float)correct_predictions / n_samples * 100);
    printf("Test - Macro F1 Score: %.6f\n", macro_f1);

    log_accuracy("Host test", n_epochs, correct_predictions, n_samples, macro_f1);
    // Giải phóng bộ nhớ
    free(Z1); free(A1); free(Z2); free(A2); free(Z3); free(A3);
}

void evaluate_random_samples(float* A0, float* Y, float* W1, float* b1, float* W2, float* b2, float* W3, float* b3,
                             int n_samples, int input_layer_nodes, int hidden_layer1_nodes, int hidden_layer2_nodes, int output_layer_nodes)
{
    printf("\n\nEvaluating Random Samples ...\n");

    int random_indices[20];
    for (int i = 0; i < 20; i++) {
        random_indices[i] = rand() % n_samples;
    }

    int true_positives[output_layer_nodes];
    int false_positives[output_layer_nodes];
    int false_negatives[output_layer_nodes];
    memset(true_positives, 0, sizeof(true_positives));
    memset(false_positives, 0, sizeof(false_positives));
    memset(false_negatives, 0, sizeof(false_negatives));

    int correct_predictions = 0;

    printf("Index\tTrue Label\tPredicted Label\tConfidence\tCheck\n");

    for (int i = 0; i < 20; i++) {
        int idx = random_indices[i];

        // Forward propagation for one sample
        float Z1[hidden_layer1_nodes], A1[hidden_layer1_nodes];
        float Z2[hidden_layer2_nodes], A2[hidden_layer2_nodes];
        float Z3[output_layer_nodes], A3[output_layer_nodes];

        matmul(&A0[idx * input_layer_nodes], W1, Z1, 1, input_layer_nodes, hidden_layer1_nodes);
        add_bias(Z1, b1, 1, hidden_layer1_nodes);
        relu(Z1, A1, 1, hidden_layer1_nodes);

        matmul(A1, W2, Z2, 1, hidden_layer1_nodes, hidden_layer2_nodes);
        add_bias(Z2, b2, 1, hidden_layer2_nodes);
        relu(Z2, A2, 1, hidden_layer2_nodes);

        matmul(A2, W3, Z3, 1, hidden_layer2_nodes, output_layer_nodes);
        add_bias(Z3, b3, 1, output_layer_nodes);
        softmax(Z3, A3, 1, output_layer_nodes);

        // Predict and evaluate
        int true_label = -1, predicted_label = -1;
        float max_prob = -1.0;

        for (int j = 0; j < output_layer_nodes; j++) {
            if (Y[idx * output_layer_nodes + j] == 1.0) {
                true_label = j;
            }
            if (A3[j] > max_prob) {
                max_prob = A3[j];
                predicted_label = j;
            }
        }

        if (true_label == predicted_label) {
            correct_predictions++;
            true_positives[true_label]++;
        } else {
            false_positives[predicted_label]++;
            false_negatives[true_label]++;
        }

        if (true_label == predicted_label) {
            printf("%d\t%d\t\t%d\t\t%.2f\t\tCorrect\n", 
               idx, true_label, predicted_label, max_prob);
        } else {
            printf("%d\t%d\t\t%d\t\t%.2f\t\tWrong\n", 
               idx, true_label, predicted_label, max_prob);
        }
    }

    // Calculate F1-score and accuracy
    float macro_f1 = 0.0;
    for (int i = 0; i < output_layer_nodes; i++) {
        float precision = 0.0;
        if (true_positives[i] + false_positives[i] > 0) {
            precision = (float)true_positives[i] / (true_positives[i] + false_positives[i]);
        }

        float recall = 0.0;
        if (true_positives[i] + false_negatives[i] > 0) {
            recall = (float)true_positives[i] / (true_positives[i] + false_negatives[i]);
        }

        float f1_score = 0.0;
        if (precision + recall > 0) {
            f1_score = 2.0 * (precision * recall) / (precision + recall);
        }
        macro_f1 += f1_score;
    }
    macro_f1 /= output_layer_nodes;

    float accuracy = (float)correct_predictions / 20 * 100;

    // Kết quả
    printf("\nRandom Samples Evaluation Results:\n");
    printf("Accuracy: %.6f%%\n", accuracy);
    printf("Macro F1 Score: %.6f\n", macro_f1);
}

int main(int argc, char** argv){
    clock_t program_start = clock();

    //===== ANN STRUCT =====//
    int n_samples = 60000, n_test_samples = 10000;
    int input_layer_nodes = 784; // number of features
    int hidden_layer1_nodes = 128;
    int hidden_layer2_nodes = 128;
    int output_layer_nodes = 10; // number of class
    //======================//
    printf("============================ Neural Network  Structure ============================\n");
    printf("Layer \t\t\tNumber of Nodes\t\tOutput Shape\t\tParameter\n");
    printf("Input Layer    \t\t    %d \t\t(%d, %d) \t\t %d\n", input_layer_nodes, n_samples, input_layer_nodes, 0);
    printf("Hidden Layer 1 \t\t    %d \t\t(%d, %d) \t\t %d\n", hidden_layer1_nodes, n_samples, hidden_layer1_nodes, input_layer_nodes * hidden_layer1_nodes + hidden_layer1_nodes);
    printf("Hidden Layer 2 \t\t    %d \t\t(%d, %d) \t\t %d\n", hidden_layer2_nodes, n_samples, hidden_layer2_nodes, hidden_layer1_nodes * hidden_layer2_nodes + hidden_layer2_nodes);
    printf("Output Layer   \t\t    %d  \t\t(%d, %d)\t\t %d\n", output_layer_nodes, n_samples, output_layer_nodes, hidden_layer2_nodes * output_layer_nodes + output_layer_nodes);
    printf("===================================================================================\n\n");

    //===== DATA =====//
    float* X_train = (float*)malloc(n_samples * input_layer_nodes * sizeof(float));
    float* Y_label = (float*)malloc(n_samples * 1 * sizeof(float));
    float* Y_train = (float*)malloc(n_samples * output_layer_nodes * sizeof(float)); //onehot

    clock_t train_data_start = clock();
    read_data("Data/X_train.txt", &X_train, n_samples, input_layer_nodes, 0);
    read_data("Data/Y_train.txt", &Y_label, n_samples, 1, 1);
    to_onehot(Y_label, Y_train, n_samples, output_layer_nodes);
    clock_t train_data_end = clock();
    printf("Train Data Processing (Read and Change label to onehot) time: %.5fs\n", (float)(train_data_end - train_data_start)/CLOCKS_PER_SEC);
    //--------------------//
    float* X_test = (float*)malloc(n_test_samples * input_layer_nodes * sizeof(float));
    float* Y_label_test = (float*)malloc(n_test_samples * 1 * sizeof(float));
    float* Y_test = (float*)malloc(n_test_samples * output_layer_nodes * sizeof(float));

    clock_t test_data_start = clock();
    read_data("Data/X_test.txt", &X_test, n_test_samples, input_layer_nodes, 0);
    read_data("Data/Y_test.txt", &Y_label_test, n_test_samples, 1, 1);
    to_onehot(Y_label_test, Y_test, n_test_samples, output_layer_nodes);
    clock_t test_data_end = clock();
    printf("Test Data Processing (Read and Change label to onehot) time: %.5fs\n", (float)(test_data_end - test_data_start)/CLOCKS_PER_SEC);
    //================//


    //===== WEIGHT =====//
    float* W1 = (float*)malloc(input_layer_nodes * hidden_layer1_nodes * sizeof(float)); // 784 x 256
    float* b1 = (float*)malloc(1 * hidden_layer1_nodes * sizeof(float)); // 1 x 256

    float* W2 = (float*)malloc(hidden_layer1_nodes * hidden_layer2_nodes * sizeof(float)); // 256 x 128
    float* b2 = (float*)malloc(1 * hidden_layer2_nodes * sizeof(float)); // 1 x 128

    float* W3 = (float*)malloc(hidden_layer2_nodes * output_layer_nodes * sizeof(float)); // 128 x 10
    float* b3 = (float*)malloc(1 * output_layer_nodes * sizeof(float)); // 1 x 10

    clock_t init_weight_start = clock();
    init_weight(W1, input_layer_nodes, hidden_layer1_nodes);
    init_weight(b1, 1, hidden_layer1_nodes);

    init_weight(W2, hidden_layer1_nodes, hidden_layer2_nodes);
    init_weight(b2, 1, hidden_layer2_nodes);

    init_weight(W3, hidden_layer2_nodes, output_layer_nodes);
    init_weight(b3, 1, output_layer_nodes);
    clock_t init_weight_end = clock();
    printf("Initialize Weight time: %.5fs\n", (float)(init_weight_end - init_weight_start)/CLOCKS_PER_SEC);
    //==================//


    //===== TRAIN =====//
    int n_epochs = 1;
    if (argc == 2)
        n_epochs = atoi(argv[1]);
    float lr = 0.01;
    train(X_train, Y_train, W1, b1, W2, b2, W3, b3,
          n_samples, input_layer_nodes, hidden_layer1_nodes, 
		  hidden_layer2_nodes, output_layer_nodes, lr, n_epochs);
    //=================//


    //===== TEST =====//
    test(X_test, Y_test, W1, b1, W2, b2, W3, b3,
        n_test_samples, input_layer_nodes, hidden_layer1_nodes, hidden_layer2_nodes, output_layer_nodes,n_epochs);

    clock_t program_end = clock();
    printf("\n\nPROGRAM TIME: %.5f\n", (float)(program_end - program_start) / CLOCKS_PER_SEC);

    if (n_epochs >=20){
        // Gọi hàm evaluate_random_samples
        evaluate_random_samples(X_test, Y_test, W1, b1, W2, b2, W3, b3,
                            n_test_samples, input_layer_nodes, hidden_layer1_nodes, hidden_layer2_nodes, output_layer_nodes);
    }

    //===== FREE =====//
    free(X_train); free(Y_label); free(Y_train);
    free(X_test); free(Y_label_test); free(Y_test);
    free(W1); free(b1);
    free(W2); free(b2);
    free(W3); free(b3);
    //================//

    return 0;
}
