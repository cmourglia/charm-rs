```charm
CounterModel :: struct {
    count: i32,
}

// *T means that the parameter is a mutable reference
increment_counter :: fn(counter: *CounterModel) {
    counter.count += 1
}

Counter :: component(counter: *CounterModel) {
    Button { 
        text: String.from(counter.count)
        on_click: increment_counter(counter)
    }
}

AppModel :: struct {
    counter: CounterModel,
}

App :: component(model: *AppModel) {
    Window {
        Counter(model.counter)
    }
}
```

One could also allow toggling the visibility of the counter

```charm
CounterModel :: struct {
    count: i32,
}

increment_counter :: fn(counter: *CounterModel) {
    counter.count += 1
}

Counter :: component(counter: *CounterModel) {
    Button {
        text: String.from(counter.count)
        on_click: increment_counter(counter)
    }
}

AppModel :: struct {
    counter: CounterModel,
    visible: bool,
}

toggle :: fn(value: *bool) {
    value.* = !value.*
}

App :: component(model: *AppModel) {
    Window {
        Button {
            text: "Toggle counter"
            on_click: toggle(&model.visible)
        }

        @if model.visible {
            Counter(model.counter)
        }
    }
}
```

Or even define multiple of them

```charm
CounterModel :: struct {
    count: i32,
}

increment_counter :: fn(counter: *CounterModel) {
    counter.count += 1
}

Counter :: component(counter: *CounterModel) {
    Button {
        text: String.from(counter.count)
        on_click: increment_counter(counter)
    }
}

AppModel :: struct {
    counters: Vec(CounterModel),
}

add_counter :: fn(v: *dyn[]CounterModel) {
    Array.push(v, CounterModel())
}

App :: component(model: *AppModel) {
    Window {
        Button {
            text: "Add counter"
            on_click: add_counter(&model.counters)
        }

        @for &counter in model.counters {
            Counter(counter)
        }
    }
}
```
