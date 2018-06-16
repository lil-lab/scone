import json
import sys
import copy

examples = open(sys.argv[1]).readlines()
domain = sys.argv[2]

def alchemy_action_sequence(prev_beakers, current_beakers):
  actions = [ ]
  for j, (old_beaker, new_beaker) in enumerate(zip(prev_beakers, current_beakers)):
    old_val = old_beaker.split(":")[1].replace("_", "")
    new_val = new_beaker.split(":")[1].replace("_", "")

    if old_val != new_val:
      if old_val == "":
        for char in old_val:
          actions.append("push " + str(j + 1) + " " + char)
      elif new_val == "":
        actions.extend(["pop " + str(j + 1) for _ in range(len(old_val))])
      else:
        for k in range(min(len(old_val), len(new_val)) + 1):
          if k == len(old_val) or k == len(new_val):
            break
          if old_val[k] != new_val[k]:
            break
        common_prefix_index = k
        actions.extend(["pop " + str(j + 1) for _ in range(len(old_val) - common_prefix_index)])
        for char in new_val[common_prefix_index:]:
          actions.append("push " + str(j + 1) + " " + char)
  return actions

def scene_action_sequence(prev_slots, current_slots):
  actions = [ ]

  for j, (old_slot, new_slot) in enumerate(zip(prev_slots, current_slots)):
    oldval = old_slot.split(":")[1]
    newval = new_slot.split(":")[1]

    old_shirt = oldval[0]
    old_hat = oldval[1]

    new_shirt = newval[0]
    new_hat = newval[1]

    changed_shirt = old_shirt != new_shirt
    changed_hat = old_hat != new_hat

    if changed_shirt and changed_hat:
      if new_shirt == "_" and new_hat == "_":
        actions.append("remove_hat " + str(j + 1))
        actions.append("remove_person " + str(j + 1))
      elif old_shirt == "_" and old_hat == "_":
        actions.append("appear_person " + str(j + 1) + " " + new_shirt)
        actions.append("appear_hat " + str(j + 1) + " " + new_hat)
    elif changed_shirt:
      actions.append("appear_person " + str(j + 1) + " " + new_shirt)
    elif changed_hat:
      if new_hat != "_" and old_hat != "_":
        actions.append("remove_hat " + str(j + 1))
      actions.append("appear_hat " + str(j + 1) + " " + new_hat)

  return actions

def tangram_shape_to_letter(shape_id):
  return chr(int(shape_id) + 65)

def tangram_action_sequence(prev_slots, current_slots):
  actions = [ ] 

  prev_slots = [item for item in prev_slots if item]
  current_slots = [item for item in current_slots if item]

  if len(prev_slots) == len(current_slots):
    ## Swapping
    figures = [ ]
    indices = [ ]

    for j, (old_slot, new_slot) in enumerate(zip(prev_slots, current_slots)):
      if old_slot != new_slot:
        figures.append(tangram_shape_to_letter(new_slot.split(":")[1]))
        indices.append(j+1)
    assert len(figures) == 2
    assert len(indices) == 2

    # Remove both figures
    actions.append("remove " + str(indices[0]))
    actions.append("remove " + str(indices[1] - 1))

    # Add them back in
    actions.append("insert " + str(indices[0]) + " " + figures[0])
    actions.append("insert " + str(indices[1]) + " " + figures[1])
  else:
    if len(prev_slots) < len(current_slots):
      assert len(prev_slots) == len(current_slots) - 1
      # Remove an item

      found = False
      for j in range(len(prev_slots)):
        if prev_slots[j] != current_slots[j]:
          new_value = tangram_shape_to_letter(current_slots[j].split(":")[1])
          actions.append("insert " + str(j+1) + " " + new_value)
          found = True
          break
      if not found:
        final_slot = current_slots[-1].split(":")
        final_idx = final_slot[0]
        final_val = tangram_shape_to_letter(final_slot[1])
        actions.append("insert " + str(final_idx) + " " + final_val)
    else:
      assert len(prev_slots) == len(current_slots) + 1
      # Insert an item

      found = False
      for j in range(len(current_slots)):
        if prev_slots[j] != current_slots[j]:
          actions.append("remove " + str(j+1))
          found = True
          break
      if not found:
        final_slot = prev_slots[-1].split(":")
        final_idx = final_slot[0]
        actions.append("remove " + str(final_idx))

  return actions

def environment_modify(environment, domain):
  if domain == "tangrams":
    slots = environment.split(" ")
    new_slots = [ ]
    for slot in slots:
      if slot:
        split = slot.split(":")
        index = split[0]
        new_val = tangram_shape_to_letter(split[1])
        new_slots.append(index + ":" + new_val)
    return " ".join(new_slots)
  else:
    return environment

seqlens = { }

annot_exs = [ ]

unique_seqs = set()
unique_actions = set()
for example in examples:
  splits = example.strip().split("\t")
  annot_ex = { "identifier" : splits[0],
               "initial_env" : environment_modify(splits[1], domain),
               "utterances": [ ]}
  instrs = splits[2:] 

  prev_env = splits[1]
  for i in range(5):
    if i * 2 + 1 < len(instrs):
      current_env = instrs[i*2+1]
    else:
      current_env = ""
    ut_ex = { "instruction" : instrs[i*2],
              "after_env" : environment_modify(current_env, domain) }

    prev_slots = prev_env.strip().split(" ")
    current_slots = current_env.strip().split(" ")

    actions = [ ]
    assert domain in { "alchemy", "scene", "tangrams"}, "domain " + domain + " not recognized"
    if domain == "alchemy":
      actions = alchemy_action_sequence(prev_slots, current_slots)
    if domain == "scene":
      actions = scene_action_sequence(prev_slots, current_slots)
    if domain == "tangrams":
      actions = tangram_action_sequence(prev_slots, current_slots)

    print(instrs[i * 2])
    print(prev_slots)
    print(current_slots)
    print(actions)
    print("")

    for action in actions:
      unique_actions.add(action)

    unique_seqs.add(" ".join(actions))

    if not len(actions) in seqlens:
      seqlens[len(actions)] = 0
    seqlens[len(actions)] += 1
    
    ut_ex["actions"] = actions

    annot_ex["utterances"].append(ut_ex)

    prev_env = current_env

  annot_exs.append(annot_ex)

print(len(unique_seqs))
print(len(unique_actions))

for length, count in seqlens.items():
  print(str(length) + "\t" + str(count) + "\t" + "{:.2f}".format(100 *float(count) / (len(annot_exs * 5))))

outfile = open(sys.argv[3], "w")
outfile.write(json.dumps(annot_exs))
outfile.close()
